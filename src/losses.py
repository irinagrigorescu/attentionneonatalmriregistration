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
# losses.py
#
##############################################################################
import torch
import torch.nn.functional as F
import numpy as np
import math
from src.units import spatial_gradient_v2


class LNCCLoss:
    """
    Local (over window) normalized cross correlation loss.

    Link: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
    """

    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):
        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class NCCLoss:
    """
    NCC loss
    """

    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, pred_img, target_img, mask=None, mask2=None, eps=1e-5):
        if mask is not None and mask2 is None:
            y_true_dm = target_img * mask
            target_img_mean = torch.sum(y_true_dm) / (torch.sum(mask) + eps)
            y_true_dm = (y_true_dm - target_img_mean) * mask

            y_pred_dm = pred_img * mask
            pred_img_mean = torch.sum(y_pred_dm) / (torch.sum(mask) + eps)
            y_pred_dm = (y_pred_dm - pred_img_mean) * mask

        elif mask is not None and mask2 is not None:
            y_true_dm = target_img * mask2
            target_img_mean = torch.sum(y_true_dm) / (torch.sum(mask2) + eps)
            y_true_dm = (y_true_dm - target_img_mean) * mask2

            y_pred_dm = pred_img * mask
            pred_img_mean = torch.sum(y_pred_dm) / (torch.sum(mask) + eps)
            y_pred_dm = (y_pred_dm - pred_img_mean) * mask

        else:
            y_true_dm = target_img - torch.mean(target_img)
            y_pred_dm = pred_img - torch.mean(pred_img)

        ncc_num = torch.sum(y_true_dm * y_pred_dm).pow(2)
        ncc_den = torch.sum(torch.pow(y_true_dm, 2)) * torch.sum(torch.pow(y_pred_dm, 2)) + eps

        # return - torch.div(ncc_num, ncc_den)
        return 1.0 - torch.div(ncc_num, ncc_den)


class EDSLoss:
    """
    EDS loss
    """

    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, pred_img, target_img, mask=None, loss=True, eps=1e-5):
        if mask is not None:
            y_true_dm = target_img * mask
            y_pred_dm = pred_img * mask
        else:
            y_true_dm = target_img
            y_pred_dm = pred_img

        # 3D
        if len(pred_img.shape) - 2 == 3:
            inshape = pred_img.shape[1:4]

            d1md2 = torch.sub(y_true_dm, y_pred_dm)
            d1md2sq = torch.matmul(d1md2.view(-1, 3, 3), d1md2.view(-1, 3, 3))

            if loss:
                return 10 * torch.mean(d1md2sq[:, 0, 0] + d1md2sq[:, 1, 1] + d1md2sq[:, 2, 2])
            else:
                return (d1md2sq[:, 0, 0] + d1md2sq[:, 1, 1] + d1md2sq[:, 2, 2]).view(-1, *inshape)

        # 2D
        elif len(pred_img.shape) - 2 == 2:
            inshape = pred_img.shape[1:3]

            d1md2 = torch.sub(y_true_dm, y_pred_dm)
            d1md2sq = torch.matmul(d1md2.view(-1, 2, 2), d1md2.view(-1, 2, 2))

            if loss:
                return 5 * torch.mean(d1md2sq[:, 0, 0] + d1md2sq[:, 1, 1])
            else:
                return (d1md2sq[:, 0, 0] + d1md2sq[:, 1, 1]).view(-1, *inshape)

        else:
            raise NotImplementedError


class DDFLoss:
    """
    DDF loss
    """

    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, displacement, loss_type='be'):
        if loss_type == 'be':
            return self.bending_energy_loss(displacement)

        elif loss_type == 'sm':
            return self.smoothness_loss(displacement)

        else:
            print("[WARNING] Unimplemented loss type. Calling bending energy loss instead.")
            return self.bending_energy_loss(displacement)

    @staticmethod
    def smoothness_loss(displacement):
        dimension = len(displacement.size())

        if dimension == 5:
            grad_x = spatial_gradient_v2(displacement, spatial_axis=0)
            grad_y = spatial_gradient_v2(displacement, spatial_axis=1)
            grad_z = spatial_gradient_v2(displacement, spatial_axis=2)

            grad_x2 = torch.pow(grad_x, 2)
            grad_y2 = torch.pow(grad_y, 2)
            grad_z2 = torch.pow(grad_z, 2)

            return 0.333 * (torch.mean(grad_x2) + torch.mean(grad_y2) + torch.mean(grad_z2))

        elif dimension == 4:
            grad_x = spatial_gradient_v2(displacement, spatial_axis=0)
            grad_y = spatial_gradient_v2(displacement, spatial_axis=1)

            grad_x2 = torch.pow(grad_x, 2)
            grad_y2 = torch.pow(grad_y, 2)

            return 0.5 * (torch.mean(grad_x2) + torch.mean(grad_y2))

        else:
            assert dimension in [1, 2, 3], "[ERROR] Unimplemented size. Given: %d" % dimension

    @staticmethod
    def bending_energy_loss(displacement):
        dimension = len(displacement.size())

        if dimension == 5:
            dTdx = spatial_gradient_v2(displacement, spatial_axis=0)
            dTdy = spatial_gradient_v2(displacement, spatial_axis=1)
            dTdz = spatial_gradient_v2(displacement, spatial_axis=2)

            dTdxx = spatial_gradient_v2(dTdx, spatial_axis=0)
            dTdyy = spatial_gradient_v2(dTdy, spatial_axis=1)
            dTdzz = spatial_gradient_v2(dTdz, spatial_axis=2)

            dTdxy = spatial_gradient_v2(dTdx, spatial_axis=1)
            dTdyz = spatial_gradient_v2(dTdy, spatial_axis=2)
            dTdxz = spatial_gradient_v2(dTdx, spatial_axis=2)

            energy = torch.mean(dTdxx * dTdxx) + torch.mean(dTdyy * dTdyy) + \
                     torch.mean(dTdzz * dTdzz) + torch.mean(2 * dTdxy * dTdxy) + \
                     torch.mean(2 * dTdxz * dTdxz) + torch.mean(2 * dTdyz * dTdyz)

            return energy

        elif dimension == 4:
            dTdx = spatial_gradient_v2(displacement, spatial_axis=0)
            dTdy = spatial_gradient_v2(displacement, spatial_axis=1)

            dTdxx = spatial_gradient_v2(dTdx, spatial_axis=0)
            dTdyy = spatial_gradient_v2(dTdy, spatial_axis=1)

            dTdxy = spatial_gradient_v2(dTdx, spatial_axis=1)

            energy = torch.mean(dTdxx * dTdxx) + torch.mean(dTdyy * dTdyy) + torch.mean(2 * dTdxy * dTdxy)

            return energy

        else:
            assert dimension in [1, 2, 3], "[ERROR] Unimplemented size. Given: %d" % dimension


class KLDLoss:
    """
    KLD loss
    """

    def __init__(self, win=None, device='cpu'):
        self.win = win
        self.device = device

    def loss(self, mu, logvar):
        mu_ = torch.flatten(mu, start_dim=1)
        logvar_ = torch.flatten(logvar, start_dim=1)

        kl_loss = -0.5 * torch.sum(1 + logvar_ - torch.pow(mu_, 2) - torch.exp(logvar_), dim=1)
        return kl_loss
