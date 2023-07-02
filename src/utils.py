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
# utils.py
#
##############################################################################
import torch
import numpy as np
import copy
from src import units
import matplotlib.pyplot as plt


# ==================================================================================================================== #
#
#  GLOBAL VARIABLES
#
# ==================================================================================================================== #
MAX_DTI = 0.003

# ==================================================================================================================== #
#
#  PLOTTING FUNCTIONS
#
# ==================================================================================================================== #
def plot_sp_ch_attn(args_, epoch_,
                    enc_ch_attn_, enc_sp_attn_, dec_ch_attn_, dec_sp_attn_,
                    extra_title_=""):
    plt.figure(figsize=(20, 20))

    for ichsp, chsptype in enumerate([enc_ch_attn_, enc_sp_attn_,
                                      dec_ch_attn_, dec_sp_attn_]):
        for ii in np.arange(0, 4):
            plt.subplot(4, 4, ii + 4 * ichsp + 1)
            slcno = chsptype[ii].shape[-1] // 2
            plt.imshow(chsptype[ii][0, 0, :, :, slcno].cpu().data.numpy(),
                       vmin=0.0, vmax=1.0, cmap='seismic')
            plt.colorbar()

            if ichsp % 2 == 0 and ii == 0:
                if ichsp == 0:
                    plt.ylabel('channel attn ENC')
                else:
                    plt.ylabel('channel attn DEC')

            if ichsp % 2 == 1 and ii == 0:
                if ichsp == 1:
                    plt.ylabel('spatial attn ENC')
                else:
                    plt.ylabel('spatial attn DEC')

    plt.savefig(args_.checkpoint_dir + '/Attention_' +
                extra_title_ + '_E' + str(epoch_ + 1) + '.png',
                dpi=100, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_attention_maps(args_, epoch_, alpha_):
    nb, nc, nx, ny, nz = alpha_.shape

    plt.figure(figsize=(16, 20))

    for iimg_, img_ in enumerate([imgcurr[id_a] for imgcurr in
                                  [alpha_[0, :, nx // 2, :, :],
                                   alpha_[0, :, :, ny // 2, :],
                                   alpha_[0, :, :, :, nz // 2]] for id_a in [0, 1]]):

        plt.subplot(3, 2, iimg_ + 1)
        plt.imshow(np.rot90(img_.cpu().data.numpy()), vmin=0, vmax=1, cmap='seismic')
        plt.xticks([])
        plt.yticks([])
        if iimg_ == 0:
            plt.title('Ch1')
        elif iimg_ == 1:
            plt.title('Ch2')

    plt.savefig(
        args_.checkpoint_dir + '/Attention_E' + str(epoch_ + 1) + '.png',
        dpi=100, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_mov_fix_moved(args_, epoch_,
                       imgs_fix_, imgs_mov_, imgs_moved_,
                       disp_field_pos_, disp_field_neg_,
                       vol_size_=(128, 128, 128), device_='cpu', extra_title_='full'):
    nr = imgs_fix_.shape[1] + 1
    nc = 4

    if nr > 4:
        plt.figure(figsize=(nr * nc, 2 * nr * nc))
    else:
        plt.figure(figsize=(2 * nr * nc, nr * nc))

    for jj in np.arange(0, imgs_fix_.shape[1]):
        # # # # # Fixed
        plt.subplot(nr, nc, nc * jj + 1)
        plt.imshow(
            np.rot90(imgs_fix_.cpu().data.numpy()[0, jj, :, :, args_.crop_depth // 2]),
            vmin=0.0, vmax=1.0, cmap='Greys_r')
        plt.xticks([])
        plt.yticks([])
        plt.title('E ' + str(epoch_))
        plt.ylabel('fix')
        plt.colorbar()

        # # # # # Moving
        plt.subplot(nr, nc, nc * jj + 2)
        plt.imshow(
            np.rot90(imgs_mov_.cpu().data.numpy()[0, jj, :, :, args_.crop_depth // 2]),
            vmin=0.0, vmax=1.0, cmap='Greys_r')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('mov')
        plt.colorbar()

        # # # # # Warped
        plt.subplot(nr, nc, nc * jj + 3)
        plt.imshow(
            np.rot90(imgs_moved_.cpu().data.numpy()[0, jj, :, :, args_.crop_depth // 2]),
            vmin=0.0, vmax=1.0, cmap='Greys_r')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('moved')
        plt.colorbar()

        # # # # # Fix-Mov
        plt.subplot(nr, nc, nc * jj + 4)
        plt.imshow(
            np.rot90(imgs_fix_.cpu().data.numpy()[0, jj, :, :, args_.crop_depth // 2]) -
            np.rot90(imgs_moved_.cpu().data.numpy()[0, jj, :, :, args_.crop_depth // 2]),
            vmin=-0.5, vmax=0.5, cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('fix-moved')
        plt.colorbar()

    plt.subplot(nr, nc, nc * imgs_fix_.shape[1] + 2)
    plt.imshow(np.rot90(np.sqrt(
        disp_field_pos_.cpu().data.numpy()[0, 0, :, :, args_.crop_depth // 2] ** 2 +
        disp_field_pos_.cpu().data.numpy()[0, 1, :, :, args_.crop_depth // 2] ** 2 +
        disp_field_pos_.cpu().data.numpy()[0, 2, :, :, args_.crop_depth // 2] ** 2)),
        vmin=0, cmap='viridis')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('disp field magnitude')

    # # # JACOBIANS
    jacobian_determinant = units.calculate_jacobian_determinant(disp_field_neg_, vol_size_, device_)
    jacobian_determinant = jacobian_determinant[:, :, 4: -4, 4: -4, 4: -4]
    neg_jacobian = len(jacobian_determinant[jacobian_determinant < 0.0])
    zero_jacobian = len(jacobian_determinant[jacobian_determinant == 0.0])
    pos_jacobian = len(jacobian_determinant[jacobian_determinant > 0.0])

    plt.subplot(nr, nc, nc * imgs_fix_.shape[1] + 3)
    cmap1 = copy.copy(plt.cm.bwr)
    cc1 = plt.imshow(np.rot90(jacobian_determinant[0, 0, :, :, args_.crop_depth // 2]), cmap=cmap1)
    cc1.cmap.set_under('cyan')
    cc1.set_clim(0.0000001, 2.0)
    plt.colorbar(extend='min')
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('jac det')
    plt.title('Jn=' + str(neg_jacobian) + '|z=' + str(zero_jacobian) + '|p=' + str(pos_jacobian))

    plt.subplot(nr, nc, nc * imgs_fix_.shape[1] + 4)
    to_plot = np.rot90(np.log(jacobian_determinant)[0, 0, :, :, args_.crop_depth // 2])
    plt.imshow(to_plot, vmin=-1, vmax=1, cmap='bwr')  # coolwarm')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('log jac det')

    plt.savefig(args_.checkpoint_dir + '/Example_' + extra_title_ + '_E' + str(epoch_ + 1) + '.png',
                dpi=100, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_losses_train(args, losses_train, title_plot):
    n_epochs_losses = len(losses_train)
    keys_losses = losses_train[0].keys()
    keys_stem_losses = [x.split('_train')[0] for x in list(keys_losses) if 'train' in x]
    keys_misc_losses = [x for x in list(keys_losses) if ('train' not in x) and ('valid' not in x)]
    keys_all_losses = keys_stem_losses + keys_misc_losses
    n_keys_losses = len(keys_all_losses)


    plt.figure(figsize=(16, 3 * n_keys_losses))
    for id_, stem_ in enumerate(keys_all_losses):
        print(stem_)

        if 'loss' in stem_:
            # PLOT LOSSES NORMAL
            plt.subplot(n_keys_losses, 2, id_*2 + 1)
            plt.fill_between(np.arange(1, n_epochs_losses),
                             [x - y for x, y in zip([np.mean(x[stem_ + '_train']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_train']) for x in losses_train[1:]])],
                             [x + y for x, y in zip([np.mean(x[stem_ + '_train']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_train']) for x in losses_train[1:]])],
                             alpha=0.2)
            plt.fill_between(np.arange(1, n_epochs_losses),
                             [x - y for x, y in zip([np.mean(x[stem_ + '_valid']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_valid']) for x in losses_train[1:]])],
                             [x + y for x, y in zip([np.mean(x[stem_ + '_valid']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_valid']) for x in losses_train[1:]])],
                             alpha=0.2)
            plt.plot(np.arange(0, n_epochs_losses),
                     [np.mean(x[stem_ + '_train']) for x in losses_train],
                     c='b', label='train')
            plt.plot(np.arange(0, n_epochs_losses),
                     [np.mean(x[stem_ + '_valid']) for x in losses_train],
                     c='r', label='valid')
            plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
            plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
            plt.legend()
            plt.xlabel('epochs')
            plt.ylabel(stem_)

            # PLOT LOSSES SEMILOGY
            plt.subplot(n_keys_losses, 2, id_*2 + 2)
            plt.fill_between(np.arange(1, n_epochs_losses),
                             [x - y for x, y in zip([np.mean(x[stem_ + '_train']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_train']) for x in losses_train[1:]])],
                             [x + y for x, y in zip([np.mean(x[stem_ + '_train']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_train']) for x in losses_train[1:]])],
                             alpha=0.2)
            plt.fill_between(np.arange(1, n_epochs_losses),
                             [x - y for x, y in zip([np.mean(x[stem_ + '_valid']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_valid']) for x in losses_train[1:]])],
                             [x + y for x, y in zip([np.mean(x[stem_ + '_valid']) for x in losses_train[1:]],
                                                    [np.std(x[stem_ + '_valid']) for x in losses_train[1:]])],
                             alpha=0.2)
            plt.semilogy(np.arange(0, n_epochs_losses),
                     [np.mean(x[stem_ + '_train']) for x in losses_train],
                     c='b', label='train')
            plt.semilogy(np.arange(0, n_epochs_losses),
                     [np.mean(x[stem_ + '_valid']) for x in losses_train],
                     c='r', label='valid')
            plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
            plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
            plt.legend()
            plt.xlabel('epochs')
            plt.ylabel(stem_)

        else:
            # PLOT LR
            plt.subplot(n_keys_losses, 2, id_ * 2 + 2)
            plt.plot(np.arange(0, n_epochs_losses),
                     [np.mean(x[stem_]) for x in losses_train], label='lr')
            plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.6)
            plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
            plt.legend()
            plt.xlabel('epochs')
            plt.ylabel(stem_)

    plt.savefig(args.results_dir + '/' + title_plot + str(n_epochs_losses) + '.png',
                dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close('all')


# ====================================================================================================================
#
#  HELPER DTI FUNCTIONS
#
# ====================================================================================================================
def from_6_to_9(data_dti, device='cpu', view=True):
    # Populate [0, 3, 4, 3, 1, 5, 4, 5, 2]
    indices_tomatrix = np.array([0, 3, 4, 3, 1, 5, 4, 5, 2])

    # data_dti = 1 x 6 x Nx x Ny x Nz
    if len(data_dti.shape) - 2 == 3:
        B, Nc, Nx, Ny, Nz = data_dti.shape
        data_dti_9 = torch.zeros(B, Nx, Ny, Nz, 9).to(device)

        for i in np.arange(0, 9):
            # Di
            data_dti_9[:, :, :, :, i] = data_dti[:, indices_tomatrix[i], ...]

        if view:
            return data_dti_9.view(B, Nx, Ny, Nz, 3, 3)
        else:
            return data_dti_9

    # data_dti = 1 x 6 x Nx x Ny
    elif len(data_dti.shape) - 2 == 2:
        B, Nc, Nx, Ny = data_dti.shape
        data_dti_9 = torch.zeros(B, Nx, Ny, 9).to(device)

        for i in np.arange(0, 9):
            # Di
            data_dti_9[:, :, :, i] = data_dti[:, indices_tomatrix[i], ...]

        if view:
            return data_dti_9.view(B, Nx, Ny, 3, 3)
        else:
            return data_dti_9

    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError


def from_6_to_9_channels(data_dti, device='cpu'):
    return from_6_to_9(data_dti, device, False)


def from_6_to_2D(data_dti, device='cpu', view=True):
    # Populate [0, 3, 3, 1]
    indices_tomatrix = np.array([0, 3, 3, 1])

    # data_dti = 1 x 6 x Nx x Ny x Nz
    if len(data_dti.shape) - 2 == 3:
        B, Nc, Nx, Ny, Nz = data_dti.shape
        data_dti_4 = torch.zeros(B, Nx, Ny, Nz, 4).to(device)

        for i in np.arange(0, 4):
            # Di
            data_dti_4[:, :, :, :, i] = data_dti[:, indices_tomatrix[i], ...]

        if view:
            return data_dti_4.view(B, Nx, Ny, Nz, 2, 2)
        else:
            return data_dti_4

    # data_dti = 1 x 6 x Nx x Ny
    elif len(data_dti.shape) - 2 == 2:
        B, Nc, Nx, Ny = data_dti.shape
        data_dti_4 = torch.zeros(B, Nx, Ny, 4).to(device)

        for i in np.arange(0, 4):
            # Di
            data_dti_4[:, :, :, i] = data_dti[:, indices_tomatrix[i], ...]

        if view:
            return data_dti_4.view(B, Nx, Ny, 2, 2)
        else:
            return data_dti_4

    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError


def from_6_to_2D_channels(data_dti, device='cpu'):
    return from_6_to_2D(data_dti, device, False)


def from_9_to_6(data_dti, device='cpu'):
    # Populate [0, 4, 8, 1, 2, 5]
    indices_totensor = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]])

    # data_dti = 1 x Nx x Ny x Nz x 3 x 3
    if len(data_dti.shape) - 1 == 5:
        B, Nx, Ny, Nz, Nc, _ = data_dti.shape
        data_dti_6 = torch.zeros(B, 6, Nx, Ny, Nz).to(device)

        for i in np.arange(0, 6):
            # Di
            data_dti_6[:, i, ...] = data_dti[:, :, :, :, indices_totensor[i][0], indices_totensor[i][1]]

        return data_dti_6

    # data_dti = 1 x Nx x Ny x 3 x 3
    elif len(data_dti.shape) - 1 == 4:
        B, Nx, Ny, Nc, _ = data_dti.shape
        data_dti_6 = torch.zeros(B, 6, Nx, Ny).to(device)

        for i in np.arange(0, 6):
            # Di
            data_dti_6[:, i, ...] = data_dti[:, :, :, indices_totensor[i][0], indices_totensor[i][1]]

        return data_dti_6

    else:
        print('Method not implemented for spatial rank != 2 or 3')
        raise NotImplementedError


# ==================================================================================================================== #
#
#  CLASS FOR ARGUMENTS
#
# ==================================================================================================================== #
class ArgumentsTrainInferenceRegistration():
    """
    Arguments for the experiments
    """
    def __init__(self,
                 n_features,
                 epochs=100,
                 decay_epoch=1,
                 batch_size=1,
                 lr=0.002,
                 gpu_ids=0,
                 crop_height=128,
                 crop_width=128,
                 crop_depth=128,
                 lamda_t2w=1.0, lamda_fa=1.0, lamda_dti=1.0,
                 lamda_lab=1.0, lamda_kld=1.0, lamda_ddf=1.0,
                 lamda_rec=5000,
                 multichdata='1000',
                 smooth=1.2,
                 loss_type='NCC',
                 validation_steps=5,
                 training=False,
                 testing=False,
                 n_channels_var=1,
                 root_dir_dHCP='/data/project/dHCP_data_str4cls/3_resampled_rig/',
                 csv_dir='/home/igr18/Work/PycharmProjects/DomainAdaptationSeg/data/',
                 train_csv='dhcp_data_train.csv',
                 valid_csv='dhcp_data_validate.csv',
                 test_csv='dhcp_data_test.csv',
                 fixed_csv='dhcp_data_train_fix.csv',
                 results_dir='/data/project/Registration/3D/results/',
                 checkpoint_dir='/data/project/Registration/3D/checkpoints/',
                 exp_name='test',
                 checkpoint_dir_reg1='/data/project/Registration/3D/checkpoints/',
                 checkpoint_dir_reg2='/data/project/Registration/3D/checkpoints/',
                 reg_net='unet3D',
                 reg_net1='unet3D',
                 reg_net2='unet3D',
                 attn_net='unet3D',
                 input_nc=1,
                 input_nc_reg1=2,
                 input_nc_reg2=2,
                 output_nc=3,
                 output_nc_reg=3,
                 is_augment=False,
                 is_halfway=False):

        self.epochs = epochs
        self.decay_epoch = decay_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.validation_steps = validation_steps
        self.gpu_ids = gpu_ids
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_depth = crop_depth
        self.exp_name = exp_name
        self.lamda_t2w = lamda_t2w
        self.lamda_fa = lamda_fa
        self.lamda_dti = lamda_dti
        self.lamda_lab = lamda_lab
        self.lamda_kld = lamda_kld
        self.lamda_ddf = lamda_ddf
        self.lamda_rec = lamda_rec
        self.multichdata = multichdata
        self.smooth = smooth
        self.loss_type = loss_type
        self.n_channels_var = n_channels_var
        self.training = training
        self.testing = testing
        self.root_dir_dHCP = root_dir_dHCP
        self.csv_dir = csv_dir
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        self.fixed_csv = fixed_csv
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir_reg1 = checkpoint_dir_reg1
        self.checkpoint_dir_reg2 = checkpoint_dir_reg2
        self.n_features = n_features
        self.reg_net = reg_net
        self.reg_net1 = reg_net1
        self.reg_net2 = reg_net2
        self.attn_net = attn_net
        self.input_nc = input_nc
        self.input_nc_reg1 = input_nc_reg1
        self.input_nc_reg2 = input_nc_reg2
        self.output_nc = output_nc
        self.output_nc_reg = output_nc_reg
        self.is_augment = is_augment
        self.is_halfway = is_halfway


# ==================================================================================================================== #
#
#  HELPER FUNCTIONS
#
# ==================================================================================================================== #
def print_networks(nets, names):
    """
    Print network parameters
    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py

    :param nets:
    :param names:
    :return:
    """
    print('    ------------Number of Parameters---------------')

    for i, net in enumerate(nets):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('    [Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))

    print('    -----------------------------------------------')


def save_checkpoint(state, save_path):
    """
    To save the checkpoint

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param state:
    :param save_path:
    :return:
    """
    torch.save(state, save_path)


def load_checkpoint(ckpt_path, map_location=None):
    """
    To load the checkpoint

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param ckpt_path:
    :param map_location:
    :return:
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('    [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def cuda(xs):
    """
    Make cuda tensor

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param xs:
    :return:
    """
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


class Sample_from_Pool(object):
    """
    To store 50 generated image in a pool and sample from it when it is full
    (Shrivastava et al’s strategy)

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py

    """
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items
