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
# units.py
#
##############################################################################
import numpy as np
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler


# ====================================================================================================================
#
#  Building blocks needed for the registration networks
#
# ====================================================================================================================
def smooth_field(dense_field, sigma, spatial_rank, device='cpu'):
    """
    Function that applies smoothing to the field
    :param dense_field: the field
    :param sigma: the standard deviation of the smoothing kernel
    :param spatial_rank: the spatial rank of the tensor
    :param device: cpu or gpu
    :return: the smoothed field
    """
    nb = dense_field.shape[0]

    kernel = get_smoothing_kernel(sigma, spatial_rank)
    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    ksize = kernel.shape[-1]

    # print('ksize', ksize, kernel.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(kernel[0,0].cpu().data.numpy())
    # plt.colorbar()
    # plt.show()

    if spatial_rank == 2:
        conv_ = partial(F.conv2d)
    elif spatial_rank == 3:
        conv_ = partial(F.conv3d)
    else:
        conv_ = partial(F.conv1d)

    if nb > 1:
        smoothed_final = []

        for ii in np.arange(0, nb):
            smoothed = [
                conv_(coord.unsqueeze(0), kernel, padding=ksize // 2)
                for coord in torch.unbind(dense_field[[ii], ...], dim=1)]

            smoothed_final.append(torch.cat(smoothed, dim=1))

        # print('smoothed_final', torch.cat(smoothed_final, dim=0).shape)

        return torch.cat(smoothed_final, dim=0)

    else:
        smoothed = [
            conv_(coord.unsqueeze(0), kernel, padding=ksize // 2)
            for coord in torch.unbind(dense_field, dim=1)]

        return torch.cat(smoothed, dim=1)


def get_smoothing_kernel(sigma, spatial_rank):
    """
    Function that creates a smoothing kernel -- written by Wenqi

    Link https://github.com/NifTK/NiftyNet/blob/dev/niftynet/network/interventional_dense_net.py

    :param sigma: standard deviation of the gaussian
    :param spatial_rank: spatial rank of the tensor
    :return: the kernel
    """
    if sigma <= 0:
        raise NotImplementedError
    tail = int(sigma * 2)
    if spatial_rank == 2:
        x, y = np.mgrid[-tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    elif spatial_rank == 3:
        x, y, z = np.mgrid[-tail:tail + 1, -tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y + z * z) / (sigma * sigma))
    else:
        raise NotImplementedError
    return g / g.sum()


def calculate_jacobian(displacement, vol_size, device='cpu'):
    """
    This function calculates Jacobian of the deformation field, given a displacement field

    :param displacement:
    :return:
    """
    if len(vol_size) == 2:
        identity_grid = get_identity_deformation_field(vol_size).to(device)

        # Make the displacement field into a deformation field
        def_field = displacement + identity_grid

        dDdx = spatial_gradient_v2(def_field, spatial_axis=0) / 2.0
        dDdy = spatial_gradient_v2(def_field, spatial_axis=1) / 2.0

        dDdx = F.pad(input=dDdx, pad=[0, 0, 1, 1], mode='constant', value=0)
        dDdy = F.pad(input=dDdy, pad=[1, 1, 0, 0], mode='constant', value=0)

        Jac = torch.cat([dDdx, dDdy], dim=1)

        return Jac

    elif len(vol_size) == 3:
        identity_grid = get_identity_deformation_field(vol_size).to(device)

        # Make the displacement field into a deformation field
        def_field = displacement + identity_grid

        dDdx = spatial_gradient_v2(def_field, spatial_axis=0) / 2.0
        dDdy = spatial_gradient_v2(def_field, spatial_axis=1) / 2.0
        dDdz = spatial_gradient_v2(def_field, spatial_axis=2) / 2.0

        dDdx = F.pad(input=dDdx, pad=(0, 0, 0, 0, 1, 1), mode='constant', value=0)
        dDdy = F.pad(input=dDdy, pad=(0, 0, 1, 1, 0, 0), mode='constant', value=0)
        dDdz = F.pad(input=dDdz, pad=(1, 1, 0, 0, 0, 0), mode='constant', value=0)

        Jac = torch.cat([dDdx, dDdy, dDdz], dim=1)

        return Jac
    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError


def calculate_jacobian_determinant(displacement, vol_size, device='cpu'):
    """
    Function that calculates the Jacobian determinant of a displacement field

    :param displacement:
    :param vol_size:
    :param device:
    :return:
    """
    nb = displacement.size()[0]
    Jac = calculate_jacobian(displacement, vol_size, device)

    if len(vol_size) == 2:
        Jac = Jac.permute(0, 2, 3, 1).view(-1, 2, 2)
    elif len(vol_size) == 3:
        Jac = Jac.permute(0, 2, 3, 4, 1).view(-1, 3, 3)
    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError

    jac_det = np.linalg.det(Jac.cpu().data.numpy())

    return jac_det.reshape((nb, 1, *vol_size))


def get_identity_deformation_field(size):
    """
    Function that creates a sampling grid
    :param size:
    :return:
    """
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)        # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor)

    return grid


def spatial_gradient_v2(img, spatial_axis):
    """
    Function that computes image spatial gradients.

    Link: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/spatial_gradient.html#SpatialGradientLayer

    Computing spatial gradient of ``input_tensor`` along
    ``self.spatial_axis``.

    output is equivalent to convolve along ``spatial_axis`` with a
     kernel: ``[-1, 0, 1]``

    This layer assumes the first and the last dimension of the input
    tensor represent batch and feature channels.
    Therefore ``spatial_axis=1`` is computing gradient along the
    third dimension of input tensor, i.e., ``input_tensor[:, :, :, y, ...]``

    Given the input with shape ``[B, C, X, Y, Z]``, and ``spatial_axis=1``
    the output shape is:  [B, C, X, Y-2, Z]

    Setting do_cropping to True makes the output tensor has the same
    dimensionality for different ``spatial_axis``.

    :param input_tensor: a batch of images with a shape of
        ``[Batch, Channel, x[, y, z, ... ]]``
    :return: spatial gradients of ``input_tensor``
    """

    spatial_rank = len(img.shape) - 2
    # print(spatial_rank)
    spatial_size = list(img.shape[2:])
    # print(spatial_size)

    # remove two elements along the gradient dim only
    spatial_size[spatial_axis] = spatial_size[spatial_axis] - 2
    spatial_begins = [0] * spatial_rank

    spatial_begins[spatial_axis] = 2
    begins_0 = [x for x in spatial_begins]

    spatial_begins[spatial_axis] = 0
    begins_1 = [x for x in spatial_begins]

    sizes_0 = spatial_size
    sizes_1 = spatial_size

    if spatial_rank == 2:
        y1 = img[:, :,
             begins_0[0]:begins_0[0] + sizes_0[0],
             begins_0[1]:begins_0[1] + sizes_0[1]]

        y2 = img[:, :,
             begins_1[0]:begins_1[0] + sizes_1[0],
             begins_1[1]:begins_1[1] + sizes_1[1]]
    elif spatial_rank == 3:
        y1 = img[:, :,
             begins_0[0]:begins_0[0] + sizes_0[0],
             begins_0[1]:begins_0[1] + sizes_0[1],
             begins_0[2]:begins_0[2] + sizes_0[2]]

        y2 = img[:, :,
             begins_1[0]:begins_1[0] + sizes_1[0],
             begins_1[1]:begins_1[1] + sizes_1[1],
             begins_1[2]:begins_1[2] + sizes_1[2]]
    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError

    return y1 - y2


# ====================================================================================================================
#
#  Building blocks needed for the CBAM attention
#
# ====================================================================================================================
class ChannelAttention(nn.Module):
    def __init__(self, n_channels, kernel_size=(128, 128)):

        super(ChannelAttention, self).__init__()

        self.kernel_size = kernel_size
        self.n_channels = n_channels

        if len(kernel_size) == 1:
            self.gap = nn.AvgPool1d(self.kernel_size)
        elif len(kernel_size) == 2:
            self.gap = nn.AvgPool2d(self.kernel_size)
        elif len(kernel_size) == 3:
            self.gap = nn.AvgPool3d(self.kernel_size)
        else:
            print('[ChannelAttention Error] Unimplemented functionality')

        if len(kernel_size) == 1:
            self.map = nn.MaxPool1d(self.kernel_size)
        elif len(kernel_size) == 2:
            self.map = nn.MaxPool2d(self.kernel_size)
        elif len(kernel_size) == 3:
            self.map = nn.MaxPool3d(self.kernel_size)
        else:
            print('[ChannelAttention Error] Unimplemented functionality')

        self.fc1 = nn.Linear(in_features=self.n_channels, out_features=self.n_channels // 2)
        self.fc2 = nn.Linear(in_features=self.n_channels // 2, out_features=self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # average pool and maxpool
        out_gap = self.gap(x)
        out_map = self.map(x)

        # mlp
        out_mlp_gap = self.fc2(self.relu(self.fc1(out_gap.view(x.shape[0], -1))))
        out_mlp_map = self.fc2(self.relu(self.fc1(out_map.view(x.shape[0], -1))))
        ch_attn = self.sigmoid(out_mlp_gap + out_mlp_map)

        # weigh the input with the channel attention
        if len(self.kernel_size) == 1:
            return x * ch_attn.view(x.shape[0], x.shape[1], 1), ch_attn.view(x.shape[0], x.shape[1], 1)
        elif len(self.kernel_size) == 2:
            return x * ch_attn.view(x.shape[0], x.shape[1], 1, 1), ch_attn.view(x.shape[0], x.shape[1], 1, 1)
        else:
            return x * ch_attn.view(x.shape[0], x.shape[1], 1, 1, 1), ch_attn.view(x.shape[0], x.shape[1], 1, 1, 1)


class AvgMaxPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False, is_3D=False):
        super(BasicConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU() if relu else None

        if is_3D:
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm3d(self.out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        else:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=7, stride=1, relu=False, is_3D=False):
        super(SpatialAttention, self).__init__()

        self.avgmaxpool = AvgMaxPool()
        self.sigmoid = nn.Sigmoid()
        padding = (kernel_size - 1) // 2

        self.conv = BasicConv(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              relu=relu, is_3D=is_3D)

    def forward(self, x):
        # Average and max pooling
        gap = self.avgmaxpool(x)

        # Convolution
        sp_attn = self.conv(gap)
        sp_attn = self.sigmoid(sp_attn)

        # Scaling
        x = x * sp_attn

        return x, sp_attn

