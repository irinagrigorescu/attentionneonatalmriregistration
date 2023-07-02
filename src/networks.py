# ============================================================================
#
#    Copyright 2020-2023 Irina Grigorescu
#    Copyright 2020-2023 King's College London
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ============================================================================

##############################################################################
#
# networks.py
#
##############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch.autograd import Variable
import numpy as np

from src.units import smooth_field, calculate_jacobian, ChannelAttention, SpatialAttention


# ==================================================================================================================== #
#
#  3D Image Registration Networks
#
#  SOME BUILDING BLOCKS FROM HERE:
#  Link: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
#
# ==================================================================================================================== #
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Code adapted from Voxelmorph
    Link: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, nsteps, device='cpu', is_half=False):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps

        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)

        self.transformer = SpatialTransformer(inshape, device=device)

        self.is_half = is_half
        self.inshape = inshape
        self.device = device

    def forward(self, vec, jac_fin=None):

        vec_half, jac_fin_half = None, None
        # vec = vec * self.scale

        for i in range(self.nsteps):

            vec = vec + self.transformer(vec, vec)

            if jac_fin is not None:
                if len(self.inshape) == 3:
                    jac_temp = torch.reshape(calculate_jacobian(vec,
                                                                self.inshape,
                                                                self.device).permute(0, 2, 3, 4, 1),
                                             (1, *self.inshape, 3, 3))
                elif len(self.inshape) == 2:
                    jac_temp = torch.reshape(calculate_jacobian(vec,
                                                                self.inshape,
                                                                self.device).permute(0, 2, 3, 1),
                                             (1, *self.inshape, 2, 2))
                else:
                    print('Method not implemented for spatial rank != 2 or 3')
                    raise NotImplementedError

                jac_fin = jac_fin @ jac_temp

            if i == self.nsteps // 2:
                vec_half = vec
                jac_fin_half = jac_fin

        if self.is_half:
            if jac_fin is not None:
                return vec, vec_half, jac_fin, jac_fin_half
            else:
                return vec, vec_half
        else:
            if jac_fin is not None:
                return vec, jac_fin
            else:
                return vec



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Code adapted from Voxelmorph
    Link: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """

    def __init__(self, size, mode='bilinear', device='cpu'):
        super().__init__()

        self.mode = mode
        self.device = device

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(self.device)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


# ====================================================================================================================
class MyKrebsRegOrigNetwork(nn.Module):
    """
    Single and multi-channel image registration network, based on the work by Krebs et al.

    The main papers are:
    1) Krebs et al., Unsupervised probabilistic deformation modeling for robust diffeomorphic registration, 2018.
       --- https://arxiv.org/abs/1804.07172

    2) Krebs et al., Learning a probabilistic model for diffeomorphic registration, 2019
       --- https://arxiv.org/abs/1812.07460

    """

    def __init__(self, in_channels=2, out_channels=2,
                 inshape=(128, 128), zsize=32, smsize=0.9,
                 is_3D=False, is_dropout=False, name='baseline2D', device='cpu'):

        super(MyKrebsRegOrigNetwork, self).__init__()

        self.name = name
        print('Network chosen is {}'.format(name))

        # other params
        self.is_3D = is_3D
        self.spatial_rank = len(inshape)
        self.device = device
        self.steps = 4
        self.is_dropout = is_dropout
        self.integrate = VecInt(inshape=inshape, nsteps=self.steps, is_half=True) if self.steps > 0 else None
        self.transformer = SpatialTransformer(size=inshape, device=self.device)

        self.inshape_s1 = tuple([x for x in inshape])
        self.inshape_s2 = tuple([x // 2 for x in inshape])
        self.inshape_s3 = tuple([x // 4 for x in inshape])
        self.inshape_bneck = tuple([x // 8 for x in inshape])

        # Get partial functions for 3D or 2D and set fully connected layers
        if self.is_3D:
            conv = partial(nn.Conv3d)
            convT = partial(nn.ConvTranspose3d)
            batchnorm = partial(nn.BatchNorm3d)
            instancenorm = partial(nn.InstanceNorm3d)

            # fully connected layers for the latent space
            self.fcmu = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1] * self.inshape_bneck[2], zsize)
            self.fclogvar = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1] * self.inshape_bneck[2], zsize)
            self.fcz = nn.Linear(zsize, self.inshape_s3[0] * self.inshape_s3[1] * self.inshape_s3[2])

            self.mode = 'trilinear'

        else:
            conv = partial(nn.Conv2d)
            convT = partial(nn.ConvTranspose2d)
            batchnorm = partial(nn.BatchNorm2d)
            instancenorm = partial(nn.InstanceNorm2d)

            # fully connected layers for the latent space
            self.fcmu = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1], zsize)
            self.fclogvar = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1], zsize)
            self.fcz = nn.Linear(zsize, self.inshape_s3[0] * self.inshape_s3[1])

            self.mode = 'bilinear'

        # activations
        self.tanh = nn.Tanh()

        # dropout
        self.drop = nn.Dropout(p=0.5, inplace=True)

        # encoder
        self.conv1 = conv(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = conv(in_channels=32, out_channels=4, kernel_size=3, stride=2, padding=1)

        # decoder
        self.uconv1 = convT(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.uconv2 = convT(in_channels=32+in_channels//2, out_channels=32,
                            kernel_size=4, stride=2, padding=1)
        self.uconv3 = convT(in_channels=32+in_channels//2, out_channels=32,
                            kernel_size=4, stride=2, padding=1)

        # velocity field
        self.vconv1 = conv(in_channels=32+in_channels//2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.vconv2 = conv(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.smoothing = smooth_field
        self.smsize = smsize

    def reparameterize_(self, mu, logvar, is_train=True):
        if is_train:
            std = logvar.mul(0.5).exp_()
            eps = Variable(torch.randn_like(std))
            return mu + torch.mul(eps, std)
        else:
            return mu  # torch.zeros_like(mu)   # mu

    def encode_(self, input):
        # encoder
        x = F.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)

        # latent space
        x = x.reshape(x.shape[0], -1)
        x_mu = self.fcmu(x)
        x_logvar = self.fclogvar(x)

        return x_mu, x_logvar

    def decode_(self, input, x_z):
        # mov, fix
        fix = input[:, :input.shape[1] // 2, ...]  # first fix
        mov = input[:, input.shape[1] // 2:, ...]  # second mov

        # Scaled mov - to be concatenated
        mov_s1 = mov
        mov_s2 = F.interpolate(mov, size=self.inshape_s2, mode=self.mode, align_corners=False)
        mov_s3 = F.interpolate(mov, size=self.inshape_s3, mode=self.mode, align_corners=False)

        # Apply the self.fcz here
        if self.is_3D:
            x_z = self.fcz(x_z).view(-1, 1, self.inshape_s3[0], self.inshape_s3[1], self.inshape_s3[2])
        else:
            x_z = self.fcz(x_z).view(-1, 1, self.inshape_s3[0], self.inshape_s3[1])

        # Decode part
        x = torch.cat([F.leaky_relu(self.uconv1(x_z), negative_slope=0.2), mov_s3], dim=1)
        x = torch.cat([F.leaky_relu(self.uconv2(x), negative_slope=0.2), mov_s2], dim=1)
        x = torch.cat([F.leaky_relu(self.uconv3(x), negative_slope=0.2), mov_s1], dim=1)

        # velocity field
        vel = F.leaky_relu(self.vconv1(x), negative_slope=0.2)
        vel = self.tanh(self.vconv2(vel))

        # smoothing is before integration s=3.9 => ksize=15
        vel = self.smoothing(dense_field=vel, sigma=self.smsize,
                             spatial_rank=self.spatial_rank, device=self.device)

        return vel


    def forward(self, input_, is_train=True):
        # encoder
        x_mu, x_logvar = self.encode_(input_)

        # re-parametrize
        x_z = self.reparameterize_(x_mu, x_logvar, is_train)

        # decoder
        vel = self.decode_(input_, x_z)

        return x_mu, x_logvar, x_z, vel


class SVDLayer(nn.Module):
    """
    Custom SVD layer
    """
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device
        self.svd_layer = torch.linalg.svd
        self.fallback_svd_layer = np.linalg.svd

    def forward(self, x, on_cpu=True):

        if on_cpu:
            x_ = x.cpu()

            u, s, vh = self.fallback_svd_layer(x_.data.numpy())
            return torch.from_numpy(u).float().to(self.device), \
                   torch.from_numpy(s).float().to(self.device), \
                   torch.from_numpy(vh).float().to(self.device)

        else:
            u, s, vh = self.svd_layer(x)  #, driver='gesvda')
            return u, s, vh


# ====================================================================================================================
class MyKrebsRegOrigNetworkAttnChSp(nn.Module):
    """
    Single and multi-channel image registration network, based on the work by Krebs et al.

    ++ with added CBAM modules

    CBAM papers here:
    Woo et al., CBAM: Convolutional Block Attention Module, 2018
       --- https://arxiv.org/abs/1807.06521

    """

    def __init__(self, in_channels=2, out_channels=2,
                 inshape=(128, 128), zsize=32, smsize=0.9,
                 is_3D=False, is_dropout=False, name='cbamAttn2D', device='cpu'):
        super(MyKrebsRegOrigNetworkAttnChSp, self).__init__()

        self.name = name
        print('Network chosen is {}'.format(name))

        # other params
        self.is_3D = is_3D
        self.spatial_rank = len(inshape)
        self.device = device
        self.steps = 4
        self.is_dropout = is_dropout
        self.integrate = VecInt(inshape=inshape, nsteps=self.steps, is_half=True) if self.steps > 0 else None
        self.transformer = SpatialTransformer(size=inshape, device=self.device)

        self.inshape_s1 = tuple([x for x in inshape])
        self.inshape_s2 = tuple([x // 2 for x in inshape])
        self.inshape_s3 = tuple([x // 4 for x in inshape])
        self.inshape_bneck = tuple([x // 8 for x in inshape])

        # Get partial functions for 3D or 2D and set fully connected layers
        if self.is_3D:
            conv = partial(nn.Conv3d)
            convT = partial(nn.ConvTranspose3d)
            batchnorm = partial(nn.BatchNorm3d)
            instancenorm = partial(nn.InstanceNorm3d)

            # fully connected layers for the latent space
            self.fcmu = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1] * self.inshape_bneck[2], zsize)
            self.fclogvar = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1] * self.inshape_bneck[2], zsize)
            self.fcz = nn.Linear(zsize, self.inshape_s3[0] * self.inshape_s3[1] * self.inshape_s3[2])

            self.mode = 'trilinear'

        else:
            conv = partial(nn.Conv2d)
            convT = partial(nn.ConvTranspose2d)
            batchnorm = partial(nn.BatchNorm2d)
            instancenorm = partial(nn.InstanceNorm2d)

            # fully connected layers for the latent space
            self.fcmu = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1], zsize)
            self.fclogvar = nn.Linear(4 * self.inshape_bneck[0] * self.inshape_bneck[1], zsize)
            self.fcz = nn.Linear(zsize, self.inshape_s3[0] * self.inshape_s3[1])

            self.mode = 'bilinear'

        # activations
        self.tanh = nn.Tanh()

        # dropout
        self.drop = nn.Dropout(p=0.5, inplace=True)

        # encoder
        self.conv1 = conv(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = conv(in_channels=32, out_channels=4, kernel_size=3, stride=2, padding=1)

        # encoder attention
        self.encchattn1 = ChannelAttention(n_channels=16, kernel_size=self.inshape_s1)
        self.encspattn1 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.encchattn2 = ChannelAttention(n_channels=32, kernel_size=self.inshape_s2)
        self.encspattn2 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.encchattn3 = ChannelAttention(n_channels=32, kernel_size=self.inshape_s3)
        self.encspattn3 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.encchattn4 = ChannelAttention(n_channels=4, kernel_size=self.inshape_bneck)
        self.encspattn4 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)

        # decoder
        self.uconv1 = convT(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.uconv2 = convT(in_channels=32 + in_channels // 2, out_channels=32,
                            kernel_size=4, stride=2, padding=1)
        self.uconv3 = convT(in_channels=32 + in_channels // 2, out_channels=32,
                            kernel_size=4, stride=2, padding=1)

        # decoder attention
        self.decchattn1 = ChannelAttention(n_channels=32 + in_channels // 2, kernel_size=self.inshape_s3)
        self.decspattn1 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.decchattn2 = ChannelAttention(n_channels=32 + in_channels // 2, kernel_size=self.inshape_s2)
        self.decspattn2 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.decchattn3 = ChannelAttention(n_channels=32 + in_channels // 2, kernel_size=self.inshape_s1)
        self.decspattn3 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)

        # velocity field
        self.vconv1 = conv(in_channels=32 + in_channels // 2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.vconv2 = conv(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.smoothing = smooth_field
        self.smsize = smsize

        # decoder attention
        self.velchattn1 = ChannelAttention(n_channels=16, kernel_size=self.inshape_s1)
        self.velspattn1 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)
        self.velchattn2 = ChannelAttention(n_channels=out_channels, kernel_size=self.inshape_s1)
        self.velspattn2 = SpatialAttention(in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False,
                                           is_3D=self.is_3D)


    def reparameterize_(self, mu, logvar, is_train=True):
        if is_train:
            std = logvar.mul(0.5).exp_()
            eps = Variable(torch.randn_like(std))
            return mu + torch.mul(eps, std)
        else:
            return mu  # torch.zeros_like(mu)   # mu


    def encode_(self, input):
        ####### encoder
        x = F.leaky_relu(self.conv1(input), negative_slope=0.2)
        x, encchattn1 = self.encchattn1(x)
        x, encspattn1 = self.encspattn1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x, encchattn2 = self.encchattn2(x)
        x, encspattn2 = self.encspattn2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x, encchattn3 = self.encchattn3(x)
        x, encspattn3 = self.encspattn3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x, encchattn4 = self.encchattn4(x)
        x, encspattn4 = self.encspattn4(x)

        # latent space
        x = x.reshape(x.shape[0], -1)
        x_mu = self.fcmu(x)
        x_logvar = self.fclogvar(x)

        return x_mu, x_logvar, (encchattn1, encchattn2, encchattn3, encchattn4), \
               (encspattn1, encspattn2, encspattn3, encspattn4)


    def decode_(self, input, x_z):
        # mov, fix
        fix = input[:, :input.shape[1] // 2, ...]  # first fix
        mov = input[:, input.shape[1] // 2:, ...]  # second mov

        # Scaled mov - to be concatenated
        mov_s1 = mov
        mov_s2 = F.interpolate(mov, size=self.inshape_s2, mode=self.mode, align_corners=False)
        mov_s3 = F.interpolate(mov, size=self.inshape_s3, mode=self.mode, align_corners=False)

        # Apply the self.fcz here
        if self.is_3D:
            x_z = self.fcz(x_z).view(-1, 1, self.inshape_s3[0], self.inshape_s3[1], self.inshape_s3[2])
        else:
            x_z = self.fcz(x_z).view(-1, 1, self.inshape_s3[0], self.inshape_s3[1])

        # Decode part
        x = torch.cat([F.leaky_relu(self.uconv1(x_z), negative_slope=0.2), mov_s3], dim=1)
        x, decchattn1 = self.decchattn1(x)
        x, decspattn1 = self.decspattn1(x)

        x = torch.cat([F.leaky_relu(self.uconv2(x), negative_slope=0.2), mov_s2], dim=1)
        x, decchattn2 = self.decchattn2(x)
        x, decspattn2 = self.decspattn2(x)

        x = torch.cat([F.leaky_relu(self.uconv3(x), negative_slope=0.2), mov_s1], dim=1)
        x, decchattn3 = self.decchattn3(x)
        x, decspattn3 = self.decspattn3(x)

        # velocity field
        vel = F.leaky_relu(self.vconv1(x), negative_slope=0.2)
        vel, velchattn1 = self.velchattn1(vel)
        vel, velspattn1 = self.velspattn1(vel)

        vel = self.vconv2(vel)
        vel, velchattn2 = self.velchattn2(vel)
        vel, velspattn2 = self.velspattn2(vel)

        vel = self.tanh(vel)

        # smoothing is before integration s=3.9 => ksize=15
        vel = self.smoothing(dense_field=vel, sigma=self.smsize,
                             spatial_rank=self.spatial_rank, device=self.device)

        return vel, (decchattn1, decchattn2, decchattn3, velchattn1, velchattn2), \
               (decspattn1, decspattn2, decspattn3, velspattn1, velspattn2)


    def forward(self, input_, is_train=True):
        # encoder
        x_mu, x_logvar, \
        (encchattn1, encchattn2, encchattn3, encchattn4), \
        (encspattn1, encspattn2, encspattn3, encspattn4) = self.encode_(input_)

        # re-parametrize
        x_z = self.reparameterize_(x_mu, x_logvar, is_train)

        # decoder
        vel, (decchattn1, decchattn2, decchattn3, velchattn1, velchattn2), \
        (decspattn1, decspattn2, decspattn3, velspattn1, velspattn2) = self.decode_(input_, x_z)

        return x_mu, x_logvar, x_z, vel, \
               ((encchattn1, encchattn2, encchattn3, encchattn4),
                (encspattn1, encspattn2, encspattn3, encspattn4)), \
               ((decchattn1, decchattn2, decchattn3, velchattn1, velchattn2), \
                (decspattn1, decspattn2, decspattn3, velspattn1, velspattn2))


# ====================================================================================================================
class MyAttentionVelNetwork(nn.Module):
    """
    Proposed attention velocity field network which produces spatial attention maps for locally weighting the
    single-channel pre-trained velocity fields

    """

    def __init__(self, in_channels=2, out_channels=2, nmodal=2,
                 inshape=(128, 128), is_3D=False, is_dropout=False, name='attnvel3D',
                 smsize=0.9, device='cpu'):

        super(MyAttentionVelNetwork, self).__init__()
        self.name = name
        print('Network chosen is {}'.format(name))

        # other params
        self.is_3D = is_3D
        self.spatial_rank = len(inshape)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smoothing = smooth_field
        self.smsize = smsize
        self.nmodal = nmodal

        # activations
        self.softmax = nn.Softmax(dim=1)

        self.inshape_s1 = tuple([x for x in inshape])
        self.inshape_s2 = tuple([x // 2 for x in inshape])
        self.inshape_s3 = tuple([x // 4 for x in inshape])

        # Get partial functions for 3D or 2D and set fully connected layers
        if self.is_3D:
            conv = partial(nn.Conv3d)
            convT = partial(nn.ConvTranspose3d)
            batchnorm = partial(nn.BatchNorm3d)
            instancenorm = partial(nn.InstanceNorm3d)

            self.mode = 'trilinear'

        else:
            conv = partial(nn.Conv2d)
            convT = partial(nn.ConvTranspose2d)
            batchnorm = partial(nn.BatchNorm2d)
            instancenorm = partial(nn.InstanceNorm2d)

            self.mode = 'bilinear'

        # Convolutional Layers
        self.conv1 = conv(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.uconv3 = conv(in_channels=64+self.nmodal, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.uconv2 = conv(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.uconv1 = conv(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Attention Maps
        self.conv_attn1 = conv(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv_attn2 = conv(in_channels=8, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, y):
        # convolutional layers
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # print('conv1', x.shape)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        # print('conv2', x.shape)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        # print('conv3    ', x.shape)
        if y is not None:
            x = torch.cat([x, y], dim=1)
        # print('torch cat', x.shape)

        x = F.interpolate(F.leaky_relu(self.uconv3(x), negative_slope=0.2), size=self.inshape_s3,
                          mode=self.mode, align_corners=False)
        # print('uconv3', x.shape)
        x = F.interpolate(F.leaky_relu(self.uconv2(x), negative_slope=0.2), size=self.inshape_s2,
                          mode=self.mode, align_corners=False)
        # print('uconv2', x.shape)
        x = F.interpolate(F.leaky_relu(self.uconv1(x), negative_slope=0.2), size=self.inshape_s1,
                          mode=self.mode, align_corners=False)
        # print('uconv1', x.shape)

        # Attention
        prealpha = F.leaky_relu(self.conv_attn1(x), negative_slope=0.2)
        alpha = F.leaky_relu(self.conv_attn2(prealpha), negative_slope=0.2)
        alpha = self.softmax(alpha)
        # print('alpha', alpha.shape)

        return alpha

