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
# models.py
#
##############################################################################
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import torchio

from src.networks import MyKrebsRegOrigNetwork, MyKrebsRegOrigNetworkAttnChSp, MyAttentionVelNetwork,\
                         SVDLayer, VecInt, SpatialTransformer
import src.units as units

from src.dataloaders import ImageRegistrationDataLoaderTorchio as ImgRegDataLoader

import src.utils as utils
import src.losses as losses


# ==================================================================================================================== #
#
#  Helper functions for networks and models
#
# ==================================================================================================================== #
def init_weights(net, is_dti=False):
    """
    Initialises the weights of a network
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param net:
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if is_dti:
                init.xavier_normal_(m.weight, gain=0.1)
            else:
                init.kaiming_normal_(m.weight, mode='fan_out')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and classname.find('Linear') != -1:
            if is_dti:
                init.xavier_normal_(m.weight, gain=0.1)
            else:
                init.kaiming_normal_(m.weight, mode='fan_out')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

        elif (classname.find('BatchNorm') != -1) or (classname.find('GroupNorm') != -1):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

    if is_dti:
        print('    >> Network initialized with xavier_normal_.')
    else:
        print('    >> Network initialized with kaiming_normal_.')

    net.apply(init_func)


def init_network(net, gpu_ids=[], is_dti=False):
    """
    Initialise network
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param net:
    :param gpu_ids:
    :return:
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, is_dti)
    return net


def set_grad(nets, requires_grad=False):
    """
    Set gradients
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param nets:
    :param requires_grad:
    :return:
    """
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def define_Net(input_nc, output_nc, n_features, net_name, vol_size, smooth=1.2,
               is_dropout=True, gpu_ids=[0], device='cpu', is_dti=False, nmodal=None):
    """
    Define the network (call the appropriate class based on the string provided)

    :param input_nc:
    :param output_nc:
    :param n_features:
    :param net_name:
    :param vol_size:
    :param smooth:
    :param is_dropout:
    :param gpu_ids:
    :param device:
    :param is_dti:
    :return:
    """

    if net_name == 'baseline2D':
        net = MyKrebsRegOrigNetwork(in_channels=input_nc, out_channels=output_nc,
                                    inshape=vol_size, zsize=n_features[0], smsize=smooth,
                                    is_3D=False, is_dropout=is_dropout, name='baseline2D', device=device)

    elif net_name == 'baseline3D':
        net = MyKrebsRegOrigNetwork(in_channels=input_nc, out_channels=output_nc,
                                    inshape=vol_size, zsize=n_features[0], smsize=smooth,
                                    is_3D=True, is_dropout=is_dropout, name='baseline3D', device=device)

    elif net_name == 'cbamAttn2D':
        net = MyKrebsRegOrigNetworkAttnChSp(in_channels=input_nc, out_channels=output_nc,
                                            inshape=vol_size, smsize=smooth,
                                            is_3D=False, is_dropout=is_dropout, name='cbamAttn2D', device=device)

    elif net_name == 'cbamAttn3D':
        net = MyKrebsRegOrigNetworkAttnChSp(in_channels=input_nc, out_channels=output_nc,
                                            inshape=vol_size, smsize=smooth,
                                            is_3D=True, is_dropout=is_dropout, name='cbamAttn3D', device=device)

    elif net_name == 'attnvel2D':
        net = MyAttentionVelNetwork(in_channels=input_nc, out_channels=output_nc,
                                    inshape=vol_size, smsize=smooth, nmodal=nmodal,
                                    is_3D=False, is_dropout=is_dropout, name='attnvel2D', device=device)

    elif net_name == 'attnvel3D':
        net = MyAttentionVelNetwork(in_channels=input_nc, out_channels=output_nc,
                                    inshape=vol_size, smsize=smooth, nmodal=nmodal,
                                    is_3D=True, is_dropout=is_dropout, name='attnvel3D', device=device)

    else:

        raise NotImplementedError('Model name [%s] is not recognized' % net_name)

    return init_network(net, gpu_ids, is_dti)


# ==================================================================================================================== #
#
#  Image registration baseline
#
# ==================================================================================================================== #
class BaselineImageRegistration(object):
    """
    Class for baseline image registration (single-/multi-channel)
    """

    # ============================================================================
    def __init__(self, args):

        # Parameters setup
        #####################################################
        self.vol_size = (args.crop_height, args.crop_width, args.crop_depth)
        self.is_dti = False  # True if args.multichdata[3] == "1" else False

        if args.gpu_ids is not None:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.integrate = VecInt(inshape=self.vol_size, nsteps=4, is_half=True, device=self.device)
        self.transform = SpatialTransformer(size=self.vol_size, device=self.device)
        self.svd_layer = SVDLayer(device=self.device)

        # Define the registration network
        #####################################################
        self.Reg = define_Net(input_nc=args.input_nc,
                              output_nc=args.output_nc,
                              n_features=args.n_features,
                              net_name=args.reg_net,
                              smooth=args.smooth,
                              vol_size=self.vol_size,
                              device=self.device,
                              is_dti=self.is_dti)

        utils.print_networks([self.Reg], ['Reg'])

        # Define Loss criterias
        #####################################################
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        if args.loss_type == "NCC":
            self.RecLoss = losses.NCCLoss(device=self.device)
        elif args.loss_type == "LNCC":
            self.RecLoss = losses.LNCCLoss(device=self.device)
        elif args.loss_type == "EDS":
            self.RecLoss = losses.EDSLoss(device=self.device)
        elif args.loss_type == "NCCEDS":
            print('Using  NCC+EDS Loss')
            self.RecLoss1 = losses.NCCLoss(device=self.device)
            self.RecLoss2 = losses.EDSLoss(device=self.device)
        else:
            print('[WARNING] Loss type not implemented. Reverting to global NCC.')
            self.RecLoss = losses.NCCLoss(device=self.device)

        self.DDFLoss = losses.DDFLoss(device=self.device)
        self.KLDLoss = losses.KLDLoss(device=self.device)

        # Optimizers
        #####################################################
        self.r_optimizer = torch.optim.Adam(self.Reg.parameters(), lr=args.lr, weight_decay=1e-5)

        self.r_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.r_optimizer,
                                                                base_lr=args.lr / 1000.0,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)

        # DATA Loaders
        #####################################################
        iregdloader_train = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata=args.multichdata,
            is_augment=args.is_augment,
            shuffle=True)

        iregdloader_valid = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata=args.multichdata,
            is_augment=False,
            shuffle=False)

        iregdloader_test = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata='1110',
            is_augment=False,
            shuffle=False,
            is_internal_capsule_data=True)

        iregdloader_train_fix = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.fixed_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata='1111',
            is_augment=False,
            shuffle=False,
            is_cgmfix=True,
            is_icfix=True)

        self.dataloaders = {
            'train': iregdloader_train.make_patches_dataloader(num_workers=12,
                                                               queue_length=32,
                                                               samples_per_volume=1,
                                                               batch_size=1),
            'valid': iregdloader_valid.make_patches_dataloader(num_workers=1,
                                                               queue_length=1,
                                                               samples_per_volume=1,
                                                               batch_size=1),
            'test': iregdloader_test.make_patches_dataloader(num_workers=1,
                                                             queue_length=1,
                                                             samples_per_volume=1,
                                                             batch_size=1),
            'fix': iregdloader_train_fix.make_patches_dataloader(num_workers=1,
                                                                 queue_length=1,
                                                                 samples_per_volume=1,
                                                                 batch_size=1)
        }

        # Get the fixed (target) data
        #####################################################
        for i, data_point in enumerate(self.dataloaders['fix']):

            # Fetch some data
            ##################################################
            self.t2w_fix = data_point['t2w'][torchio.DATA][None]
            self.fa_fix = data_point['fa'][torchio.DATA][None]
            self.lab_fix = data_point['lab'][torchio.DATA][None]
            self.cgm_fix = data_point['cgmfix'][torchio.DATA][None]
            self.ic_fix = data_point['icfix'][torchio.DATA][None]

            # DTI
            if args.multichdata[3] == '1':
                self.dti_fix = data_point['dti'][torchio.DATA][None]
                self.dti_fix[torch.isnan(self.dti_fix)] = 0.0

                self.dti_fix = utils.from_6_to_9_channels(self.dti_fix).permute(0, 4, 1, 2, 3)

                self.dti_mask = torch.zeros_like(self.dti_fix[:, [0], ...])
                self.dti_mask[:, :, 4:-4, 4:-4, 4:-4] = 1.0
                self.dti_mask = self.dti_mask.permute(0, 2, 3, 4, 1)
                self.dti_mask = torch.reshape(torch.tile(self.dti_mask, (1, 1, 1, 1, 9)),
                                     (*self.dti_mask.shape[:-1], 3, 3)).float().to(self.device)
                self.dti_mask_inv = torch.abs(1. - self.dti_mask)

                self.jac_ini = torch.tile(torch.reshape(torch.eye(3), (1, 1, 1, 1, 3, 3)),
                                          (1, *self.vol_size, 1, 1)).to(self.device)

            else:
                self.dti_fix = None

            self.as_fix = data_point['AS']
            self.aff_fix = data_point['t2w']['affine']

            break

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Reg.load_state_dict(ckpt['Reg'])
            self.r_optimizer.load_state_dict(ckpt['r_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_reconstruction_loss = 1e10
        plot_step = 5
        img_input_once = None

        # Create the fixed images only once for speed
        fixed_images, loss_weights = [], []
        flags = {'t2w': None, 'cgm': None, 'lab': None, 'dti': None, 'fa': None}

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.r_optimizer.param_groups[0]['lr']
            print('LEARNING RATE = %.7f' % lr)

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'rec_loss_train': [], 'rec_loss_valid': [],
                       'kld_loss_train': [], 'kld_loss_valid': [],
                       'ddf_loss_train': [], 'ddf_loss_valid': [],
                       'total_loss_train': [], 'total_loss_valid': [],
                       'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # For each epoch set the validation losses to 0
            rec_loss_valid = 0.0
            ddf_loss_valid = 0.0
            kld_loss_valid = 0.0

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train', 'valid']:

                for i, data_point in enumerate(self.dataloaders[phase]):

                    if i > 1 :
                        break

                    # step
                    len_dataloader = len(self.dataloaders[phase])
                    step = epoch * len_dataloader + i + 1

                    # Fetch data T2w|cGM|FA|DTI
                    ##################################################
                    moving_images = []

                    # T2W
                    if args.multichdata[0] == '1':
                        t2w_mov = utils.cuda(Variable(data_point['t2w'][torchio.DATA][None]))
                        moving_images.append(t2w_mov)

                        if flags['t2w'] is None:
                            t2w_fix = utils.cuda(Variable(torch.zeros_like(self.t2w_fix) + self.t2w_fix))
                            fixed_images.append(t2w_fix)
                            loss_weights.append(args.lamda_t2w)
                            flags['t2w'] = 'ok'

                    # CGM
                    if args.multichdata[1] == '1':
                        lab_mov = data_point['lab'][torchio.DATA][None]
                        cgm_mov = torch.zeros_like(lab_mov)
                        cgm_mov[lab_mov == 2] = 1.0

                        cgm_mov = utils.cuda(Variable(cgm_mov))
                        moving_images.append(cgm_mov)

                        if flags['cgm'] is None:
                            cgm_fix = utils.cuda(Variable(torch.zeros_like(self.cgm_fix) + self.cgm_fix))
                            fixed_images.append(cgm_fix)
                            loss_weights.append(args.lamda_lab)
                            flags['cgm'] = 'ok'

                    # FA
                    if args.multichdata[2] == '1':
                        fa_mov = utils.cuda(Variable(data_point['fa'][torchio.DATA][None]))
                        moving_images.append(fa_mov)

                        if flags['fa'] is None:
                            fa_fix = utils.cuda(Variable(torch.zeros_like(self.fa_fix) + self.fa_fix))
                            fixed_images.append(fa_fix)
                            loss_weights.append(args.lamda_fa)
                            flags['fa'] = 'ok'

                    # DTI
                    if args.multichdata[3] == '1':
                        dti_mov = data_point['dti'][torchio.DATA][None]
                        dti_mov[torch.isnan(dti_mov)] = 0.0
                        dti_mov = utils.from_6_to_9_channels(dti_mov).permute(0, 4, 1, 2, 3)
                        moving_images.append(utils.cuda(Variable(dti_mov)))

                        if flags['dti'] is None:
                            dti_fix = utils.cuda(Variable(torch.zeros_like(self.dti_fix) + self.dti_fix))
                            fixed_images.append(dti_fix)
                            loss_weights.append(args.lamda_dti)
                            flags['dti'] = 'ok'


                    ##################################################
                    # PREPARE INPUT DATA
                    ##################################################
                    img_input = Variable(torch.cat(fixed_images + moving_images, dim=1).float()).to(self.device)
                    nmodalities = img_input.shape[1] // 2

                    # TODO: delete this it's just for test
                    if epoch == self.start_epoch and i == 0 and phase == 'valid' and img_input_once is None:
                        img_input_once = utils.cuda(Variable(torch.zeros_like(img_input) + img_input))

                    # TRAIN
                    ##################################################
                    if phase == 'train':
                        self.Reg.train()

                        # Forward pass through registration network
                        ##################################################
                        if self.Reg.module.name == "baseline2D" or self.Reg.module.name == "baseline3D":
                            x_mu, x_logvar, x_z, vel = self.Reg(img_input)

                        elif self.Reg.module.name == "cbamAttn2D" or self.Reg.module.name == "cbamAttn3D":
                            x_mu, x_logvar, x_z, vel, \
                            (enc_ch_attn, enc_sp_attn), (dec_ch_attn, dec_sp_attn) = self.Reg(img_input)

                        else:
                            print("[WARNING] The module specified does not exist")
                            print("Try one of : baseline2D baseline3D cbamAttn2D cbamAttn3D")

                        # Do a train loop
                        ##################################################
                        (rec_loss, kld_loss, ddf_loss, total_loss), _, _ = \
                               self.train_validate_loop(args, img_input, vel, x_mu, x_logvar, nmodalities, loss_weights)

                        metrics['rec_loss_train'].append(rec_loss.item())
                        metrics['kld_loss_train'].append(kld_loss.item())
                        metrics['ddf_loss_train'].append(ddf_loss.item())
                        metrics['total_loss_train'].append(total_loss.item())

                        # Update network
                        ###################################################
                        self.r_optimizer.zero_grad()
                        total_loss.backward()
                        self.r_optimizer.step()


                    # VALIDATE
                    #######################################################
                    else:
                        img_input = utils.cuda(Variable(torch.zeros_like(img_input_once) + img_input_once))

                        self.Reg.eval()

                        with torch.no_grad():
                            # Forward pass through registration network
                            ##################################################
                            if self.Reg.module.name == "baseline2D" or self.Reg.module.name == "baseline3D":
                                x_mu, x_logvar, x_z, vel = self.Reg(img_input)

                            elif self.Reg.module.name == "cbamAttn2D" or self.Reg.module.name == "cbamAttn3D":
                                x_mu, x_logvar, x_z, vel, \
                                (enc_ch_attn, enc_sp_attn), (dec_ch_attn, dec_sp_attn) = self.Reg(img_input)

                            else:
                                print("[WARNING] The module specified does not exist")
                                print("Try one of : baseline2D baseline3D cbamAttn2D cbamAttn3D")

                        # Do a validation loop
                        ##################################################
                        (rec_loss, kld_loss, ddf_loss, total_loss), \
                            (warpfull_mov, disp_field_pos, disp_field_neg), \
                            (warphalf_mov, warphalf_fix, disp_half_pos, disp_half_neg) = \
                               self.train_validate_loop(args, img_input, vel, x_mu, x_logvar, nmodalities, loss_weights)

                        metrics['rec_loss_valid'].append(rec_loss.item())
                        metrics['kld_loss_valid'].append(kld_loss.item())
                        metrics['ddf_loss_valid'].append(ddf_loss.item())
                        metrics['total_loss_valid'].append(total_loss.item())

                        # Save valid losses here:
                        rec_loss_valid += rec_loss.item()
                        ddf_loss_valid += ddf_loss.item()
                        kld_loss_valid += kld_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True

                            utils.plot_mov_fix_moved(args, epoch,
                                               img_input[:, :nmodalities, ...],
                                               img_input[:, nmodalities:, ...],
                                               warpfull_mov,
                                               disp_field_pos, disp_field_neg,
                                               vol_size_=self.vol_size, device_=self.device, extra_title_='full')

                            if args.is_halfway:
                                utils.plot_mov_fix_moved(args, epoch,
                                                         img_input[:, :nmodalities, ...],
                                                         warphalf_mov, warphalf_fix,
                                                         disp_half_pos, disp_half_neg,
                                                         vol_size_=self.vol_size, device_=self.device,
                                                         extra_title_='half')

                            if self.Reg.module.name == "cbamAttn2D" or self.Reg.module.name == "cbamAttn3D":

                                utils.plot_sp_ch_attn(args, epoch,
                                                      enc_ch_attn, enc_sp_attn, dec_ch_attn, dec_sp_attn)


                        # Save best after all validation steps
                        #######################################################
                        if i >= (args.validation_steps - 1):
                            rec_loss_valid /= args.validation_steps
                            ddf_loss_valid /= args.validation_steps
                            kld_loss_valid /= args.validation_steps

                            print('AVG REC LOSS VALID | ', rec_loss_valid)
                            print('AVG DDF LOSS VALID | ', ddf_loss_valid)
                            print('AVG KLD LOSS VALID | ', kld_loss_valid)

                            # Save best
                            if best_reconstruction_loss > rec_loss_valid and epoch > 0:
                                best_reconstruction_loss = rec_loss_valid
                                print("Best Reconstruction Valid Loss %.2f" % (best_reconstruction_loss))

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Reg': self.Reg.state_dict(),
                                                       'r_optimizer': self.r_optimizer.state_dict(),
                                                       'losses_train': self.losses_train},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Rec Loss: %f' % (epoch + 1, rec_loss_valid))

                            # Stop early -- Don't go through all the validation set
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print(
                        "%s Epoch: (%3d) (%5d/%5d) (%3d) | Total Loss: %.2e | Rec Loss:%.2e | KLD Loss:%.2e | DDF Loss:%.2e | %.0fm %.2fs" %
                        (phase.upper(), epoch, i + 1, len_dataloader, step,
                         total_loss.item(), rec_loss.item(), kld_loss.item(), ddf_loss.item(), time_elapsed // 60,
                         time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Reg': self.Reg.state_dict(),
                                   'r_optimizer': self.r_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.r_lr_scheduler.step()

        return self.losses_train


    def train_validate_loop(self, args_, img_input_, vel_, x_mu_, x_logvar_, nmodalities_, loss_weights_):

        # Transform vel_ field into displacement field
        ##################################################
        flow_pos, flow_neg = vel_, -vel_

        # Scaling and squaring
        disp_field_pos_, disp_half_pos_ = self.integrate(flow_pos)
        disp_field_neg_, disp_half_neg_ = self.integrate(flow_neg)

        # Warp the moving all the way
        ##################################################
        warpfull_mov_ = self.transform(img_input_[:, nmodalities_:, ...], disp_field_pos_)

        # and the moving and fixed halfway (this is more time consuming)
        #############################
        if args_.is_halfway:
            warphalf_mov_ = self.transform(img_input_[:, nmodalities_:, ...], disp_half_pos_)
            warphalf_fix_ = self.transform(img_input_[:, :nmodalities_, ...], disp_half_neg_)

        # DTI is treated as a special case
        ##################################################
        if args_.multichdata[3] == '1':  # DTI is a channel

            # Calculate jacobian from the final displacement field directly
            ##################################################
            disp_field_pos_jac = units.calculate_jacobian(disp_field_pos_, self.vol_size, self.device)
            disp_field_pos_jac = disp_field_pos_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

            # Mask edges of jacobian
            ##################################################
            if self.dti_mask is not None:
                disp_field_pos_jac = (disp_field_pos_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)

            # Calculate rotation matrix from Jacobian doing polar decomposition
            ##################################################
            ## Note: Bare in mind that torch svd has the tendency of being extremely slow
            #        and I have empirically found that using the CPU version is much better
            u, _, vh = self.svd_layer(disp_field_pos_jac)
            R_pos = torch.matmul(u, vh)
            R_pos_tr = torch.transpose(R_pos, dim0=4, dim1=5)

            if args_.is_halfway:
                # Calculate jacobian from the halfway positive displacement
                ##################################################
                disp_field_half_pos_jac = units.calculate_jacobian(disp_half_pos_, self.vol_size, self.device)
                disp_field_half_pos_jac = disp_field_half_pos_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

                # Calculate jacobian from the halfway negative displacement
                ##################################################
                disp_field_half_neg_jac = units.calculate_jacobian(disp_half_neg_, self.vol_size, self.device)
                disp_field_half_neg_jac = disp_field_half_neg_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

                # Mask edges of jacobian
                ##################################################
                if self.dti_mask is not None:
                    disp_field_half_pos_jac = (disp_field_half_pos_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)
                    disp_field_half_neg_jac = (disp_field_half_neg_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)

                # Calculate rotation matrix from Jacobian halfway pos doing polar decomposition
                ##################################################
                u, _, vh = self.svd_layer(disp_field_half_pos_jac)
                R_pos_half_pos = torch.matmul(u, vh)
                R_pos_tr_half_pos = torch.transpose(R_pos, dim0=4, dim1=5)

                # Calculate rotation matrix from Jacobian halfway neg doing polar decomposition
                ##################################################
                u, _, vh = self.svd_layer(disp_field_half_neg_jac)
                R_pos_half_neg = torch.matmul(u, vh)
                R_pos_tr_half_neg = torch.transpose(R_pos, dim0=4, dim1=5)

            # Reorient the tensors ONLY
            ##################################################
            # case 1 channel == DTI-only
            if len(loss_weights_) == 1:
                warpfull_mov_dti = torch.matmul(
                    torch.matmul(R_pos,
                                 warpfull_mov_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                             *self.vol_size,
                                                                             3, 3)),
                    R_pos_tr)

                if args_.is_halfway:
                    warphalf_mov_dti = torch.matmul(
                        torch.matmul(R_pos_half_pos,
                                     warphalf_mov_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                                 *self.vol_size,
                                                                                 3, 3)),
                        R_pos_tr_half_pos)

                    warphalf_fix_dti = torch.matmul(
                        torch.matmul(R_pos_half_neg,
                                     warphalf_fix_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                                 *self.vol_size,
                                                                                 3, 3)),
                        R_pos_tr_half_neg)

                    warphalf_mov_dti = warphalf_mov_dti.view(1, *self.vol_size, 9)
                    warphalf_fix_dti = warphalf_fix_dti.view(1, *self.vol_size, 9)

            # case 2 channels == T2w + DTI (scalar + dti)
            else:  # len(loss_weights_) == 2:
                warpfull_mov_dti = torch.matmul(
                    torch.matmul(R_pos,
                                 warpfull_mov_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                         *self.vol_size,
                                                                                         3, 3)),
                    R_pos_tr)

                if args_.is_halfway:
                    warphalf_mov_dti = torch.matmul(
                        torch.matmul(R_pos_half_pos,
                                     warphalf_mov_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                             *self.vol_size,
                                                                                             3, 3)),
                        R_pos_tr_half_pos)

                    warphalf_fix_dti = torch.matmul(
                        torch.matmul(R_pos_half_neg,
                                     warphalf_fix_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                             *self.vol_size,
                                                                                             3, 3)),
                        R_pos_tr_half_neg)

                    warphalf_mov_dti = warphalf_mov_dti.view(1, *self.vol_size, 9)
                    warphalf_fix_dti = warphalf_fix_dti.view(1, *self.vol_size, 9)

            warpfull_mov_dti = warpfull_mov_dti.view(1, *self.vol_size, 9)

            # Reconstruction Loss
            ###################################################
            rec_loss_ = 0

            # dti only
            if len(loss_weights_) == 1 and args_.multichdata[3] == '1':
                rec_loss_ += loss_weights_[0] * \
                            (self.RecLoss.loss(warpfull_mov_dti,
                                               img_input_[:, :nmodalities_, ...].permute(0, 2, 3, 4, 1)))
                if args_.is_halfway:
                    rec_loss_ += loss_weights_[0] * self.RecLoss.loss(warphalf_fix_dti, warphalf_mov_dti)

            # t2w + dti
            elif len(loss_weights_) == 2 and args_.multichdata[3] == '1':
                rec_loss_ += loss_weights_[0] * \
                            (self.RecLoss1.loss(warpfull_mov_[:, :1, ...],
                                                img_input_[:, :1, ...]))
                rec_loss_ += loss_weights_[1] * \
                            (self.RecLoss2.loss(warpfull_mov_dti,
                                                img_input_[:, 1:nmodalities_, ...].permute(0, 2, 3, 4, 1)))

                if args_.is_halfway:
                    rec_loss_ += loss_weights_[0] * \
                                self.RecLoss1.loss(warphalf_fix_[:, :1, ...], warphalf_mov_[:, :1, ...])

                    rec_loss_ += loss_weights_[1] * \
                                self.RecLoss2.loss(warphalf_fix_dti, warphalf_mov_dti)

        else:
            # Reconstruction Loss
            ###################################################
            rec_loss_ = 0

            for ilamda_, lamda_ in enumerate(loss_weights_):
                rec_loss_ += lamda_ * self.RecLoss.loss(img_input_[:, ilamda_:ilamda_ + 1, ...],
                                                       warpfull_mov_[:, ilamda_:ilamda_ + 1, ...])

                if args_.is_halfway:
                    rec_loss_ += lamda_ * self.RecLoss.loss(warphalf_fix_[:, ilamda_:ilamda_ + 1, ...],
                                                            warphalf_mov_[:, ilamda_:ilamda_ + 1, ...])

        # Smoothness Loss
        ###################################################
        ddf_loss_ = self.DDFLoss.loss(disp_field_pos_, loss_type='be') * args_.lamda_ddf

        # KLD Loss
        ###################################################
        kld_loss_ = self.KLDLoss.loss(x_mu_, x_logvar_) * args_.lamda_kld

        # Total Loss
        ###################################################
        total_loss_ = args_.lamda_rec * (rec_loss_ + ddf_loss_) + kld_loss_

        # Return losses and intermediate values
        ###################################################
        if args_.is_halfway:
            return (rec_loss_, kld_loss_, ddf_loss_, total_loss_),\
                   (warpfull_mov_, disp_field_pos_, disp_field_neg_), \
                   (warphalf_mov_, warphalf_fix_, disp_half_pos_, disp_half_neg_)
        else:
            return (rec_loss_, kld_loss_, ddf_loss_, total_loss_), \
                   (warpfull_mov_, disp_field_pos_, disp_field_neg_), \
                   (None, None, None, None)



# ==================================================================================================================== #
#
#  Image registration with attention on the velocity fields
#
# ==================================================================================================================== #
class AttentionVelFieldImageRegistration(object):
    """
    Class for baseline image registration (single-/multi-channel)
    """

    # ============================================================================
    def __init__(self, args):

        # Parameters setup
        #####################################################
        self.vol_size = (args.crop_height, args.crop_width, args.crop_depth)
        self.is_dti = True if args.multichdata[3] == "1" else False

        if args.gpu_ids is not None:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.integrate = VecInt(inshape=self.vol_size, nsteps=4, is_half=True, device=self.device)
        self.transform = SpatialTransformer(size=self.vol_size, device=self.device)
        self.svd_layer = SVDLayer(device=self.device)

        # Define the registration networks (pre-trained)
        #####################################################
        self.Reg1 = define_Net(input_nc=args.input_nc_reg1,
                               output_nc=args.output_nc_reg,
                               n_features=args.n_features,
                               net_name=args.reg_net1,
                               smooth=args.smooth,
                               vol_size=self.vol_size,
                               device=self.device,
                               is_dti=self.is_dti)

        self.Reg2 = define_Net(input_nc=args.input_nc_reg2,
                               output_nc=args.output_nc_reg,
                               n_features=args.n_features,
                               net_name=args.reg_net2,
                               smooth=args.smooth,
                               vol_size=self.vol_size,
                               device=self.device,
                               is_dti=self.is_dti)

        self.Attn = define_Net(input_nc=args.input_nc,
                               output_nc=args.output_nc,
                               n_features=args.n_features,
                               net_name=args.attn_net,
                               smooth=args.smooth,
                               vol_size=self.vol_size,
                               device=self.device,
                               nmodal=(args.input_nc_reg1 + args.input_nc_reg2) // 2)

        utils.print_networks([self.Reg1, self.Reg2, self.Attn], ['Reg1', 'Reg2', 'Attn'])

        # Define Loss criterias
        #####################################################
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        if args.loss_type == "NCC":
            self.RecLoss = losses.NCCLoss(device=self.device)
        elif args.loss_type == "LNCC":
            self.RecLoss = losses.LNCCLoss(device=self.device)
        elif args.loss_type == "EDS":
            self.RecLoss = losses.EDSLoss(device=self.device)
        elif args.loss_type == "NCCEDS":
            print('Using  NCC+EDS Loss')
            self.RecLoss1 = losses.NCCLoss(device=self.device)
            self.RecLoss2 = losses.EDSLoss(device=self.device)
        else:
            print('[WARNING] Loss type not implemented. Reverting to global NCC.')
            self.RecLoss = losses.NCCLoss(device=self.device)

        self.DDFLoss = losses.DDFLoss(device=self.device)
        self.KLDLoss = losses.KLDLoss(device=self.device)

        # Optimizers
        #####################################################
        self.r_optimizer = torch.optim.Adam(self.Attn.parameters(), lr=args.lr, weight_decay=1e-5)

        self.r_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.r_optimizer,
                                                                base_lr=args.lr / 1000.0,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)

        # DATA Loaders
        #####################################################
        iregdloader_train = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata=args.multichdata,
            is_augment=args.is_augment,
            shuffle=True)

        iregdloader_valid = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata=args.multichdata,
            is_augment=False,
            shuffle=False)

        iregdloader_test = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata='1110',
            is_augment=False,
            shuffle=False,
            is_internal_capsule_data=True)

        iregdloader_train_fix = ImgRegDataLoader(
            csv_file=os.path.join(args.csv_dir, args.fixed_csv),
            root_dir=args.root_dir_dHCP,
            output_size=self.vol_size,
            multichdata='1111',
            is_augment=False,
            shuffle=False,
            is_cgmfix=True,
            is_icfix=True)

        self.dataloaders = {
            'train': iregdloader_train.make_patches_dataloader(num_workers=12,
                                                               queue_length=32,
                                                               samples_per_volume=1,
                                                               batch_size=1),
            'valid': iregdloader_valid.make_patches_dataloader(num_workers=1,
                                                               queue_length=1,
                                                               samples_per_volume=1,
                                                               batch_size=1),
            'test': iregdloader_test.make_patches_dataloader(num_workers=1,
                                                             queue_length=1,
                                                             samples_per_volume=1,
                                                             batch_size=1),
            'fix': iregdloader_train_fix.make_patches_dataloader(num_workers=1,
                                                                 queue_length=1,
                                                                 samples_per_volume=1,
                                                                 batch_size=1)
        }

        # Get the fixed (target) data
        #####################################################
        for i, data_point in enumerate(self.dataloaders['fix']):

            # Fetch some data
            ##################################################
            self.t2w_fix = data_point['t2w'][torchio.DATA][None]
            self.fa_fix = data_point['fa'][torchio.DATA][None]
            self.lab_fix = data_point['lab'][torchio.DATA][None]
            self.cgm_fix = data_point['cgmfix'][torchio.DATA][None]
            self.ic_fix = data_point['icfix'][torchio.DATA][None]

            # DTI
            if args.multichdata[3] == '1':
                self.dti_fix = data_point['dti'][torchio.DATA][None]
                self.dti_fix[torch.isnan(self.dti_fix)] = 0.0

                self.dti_fix = utils.from_6_to_9_channels(self.dti_fix).permute(0, 4, 1, 2, 3)

                self.dti_mask = torch.zeros_like(self.dti_fix[:, [0], ...])
                self.dti_mask[:, :, 4:-4, 4:-4, 4:-4] = 1.0
                self.dti_mask = self.dti_mask.permute(0, 2, 3, 4, 1)
                self.dti_mask = torch.reshape(torch.tile(self.dti_mask, (1, 1, 1, 1, 9)),
                                     (*self.dti_mask.shape[:-1], 3, 3)).float().to(self.device)
                self.dti_mask_inv = torch.abs(1. - self.dti_mask)

                self.jac_ini = torch.tile(torch.reshape(torch.eye(3), (1, 1, 1, 1, 3, 3)),
                                          (1, *self.vol_size, 1, 1)).to(self.device)

            else:
                self.dti_fix = None

            self.as_fix = data_point['AS']
            self.aff_fix = data_point['t2w']['affine']

            break

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoints for the pre-trained image registration networks
        #####################################################
        try:
            ckpt = utils.load_checkpoint(args.checkpoint_dir_reg1)
            self.Reg1.load_state_dict(ckpt['Reg'])
            print('        [*] Successfully loaded reg for T2w (best from before)')
        except:
            print(' [*] No checkpoint for reg 1!')

        try:
            ckpt = utils.load_checkpoint(args.checkpoint_dir_reg2)
            self.Reg2.load_state_dict(ckpt['Reg'])

            if args.multichdata[3] == '1':
                print('        [*] Successfully loaded reg for DTI (best from before)')
            else:
                print('        [*] Successfully loaded reg for FA  (best from before)')
        except:
            print(' [*] No checkpoint for reg 2!')

        print("\n\n")

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Attn.load_state_dict(ckpt['Attn'])
            self.r_optimizer.load_state_dict(ckpt['r_optimizer'])
        except:
            print(' [*] No checkpoint for attention network!')
            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_reconstruction_loss = 1e10
        plot_step = 5
        img_input_once = None

        # Create the fixed images only once for speed
        fixed_images, loss_weights = [], []
        flags = {'t2w': None, 'cgm': None, 'lab': None, 'dti': None, 'fa': None}

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.r_optimizer.param_groups[0]['lr']
            print('LEARNING RATE = %.7f' % lr)

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'rec_loss_train': [], 'rec_loss_valid': [],
                       'ddf_loss_train': [], 'ddf_loss_valid': [],
                       'total_loss_train': [], 'total_loss_valid': [],
                       'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # For each epoch set the validation losses to 0
            rec_loss_valid = 0.0
            ddf_loss_valid = 0.0
            kld_loss_valid = 0.0

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train', 'valid']:

                for i, data_point in enumerate(self.dataloaders[phase]):

                    if i > 1 :
                        break

                    # step
                    len_dataloader = len(self.dataloaders[phase])
                    step = epoch * len_dataloader + i + 1

                    # Fetch data T2w|cGM|FA|DTI
                    ##################################################
                    moving_images = []

                    # T2W
                    if args.multichdata[0] == '1':
                        t2w_mov = utils.cuda(Variable(data_point['t2w'][torchio.DATA][None]))
                        moving_images.append(t2w_mov)

                        if flags['t2w'] is None:
                            t2w_fix = utils.cuda(Variable(torch.zeros_like(self.t2w_fix) + self.t2w_fix))
                            fixed_images.append(t2w_fix)
                            loss_weights.append(args.lamda_t2w)
                            flags['t2w'] = 'ok'

                    # CGM
                    if args.multichdata[1] == '1':
                        lab_mov = data_point['lab'][torchio.DATA][None]
                        cgm_mov = torch.zeros_like(lab_mov)
                        cgm_mov[lab_mov == 2] = 1.0

                        cgm_mov = utils.cuda(Variable(cgm_mov))
                        moving_images.append(cgm_mov)

                        if flags['cgm'] is None:
                            cgm_fix = utils.cuda(Variable(torch.zeros_like(self.cgm_fix) + self.cgm_fix))
                            fixed_images.append(cgm_fix)
                            loss_weights.append(args.lamda_lab)
                            flags['cgm'] = 'ok'

                    # FA
                    if args.multichdata[2] == '1':
                        fa_mov = utils.cuda(Variable(data_point['fa'][torchio.DATA][None]))
                        moving_images.append(fa_mov)

                        if flags['fa'] is None:
                            fa_fix = utils.cuda(Variable(torch.zeros_like(self.fa_fix) + self.fa_fix))
                            fixed_images.append(fa_fix)
                            loss_weights.append(args.lamda_fa)
                            flags['fa'] = 'ok'

                    # DTI
                    if args.multichdata[3] == '1':
                        dti_mov = data_point['dti'][torchio.DATA][None]
                        dti_mov[torch.isnan(dti_mov)] = 0.0
                        dti_mov = utils.from_6_to_9_channels(dti_mov).permute(0, 4, 1, 2, 3)
                        moving_images.append(utils.cuda(Variable(dti_mov)))

                        if flags['dti'] is None:
                            dti_fix = utils.cuda(Variable(torch.zeros_like(self.dti_fix) + self.dti_fix))
                            fixed_images.append(dti_fix)
                            loss_weights.append(args.lamda_dti)
                            flags['dti'] = 'ok'


                    ##################################################
                    # PREPARE INPUT DATA
                    ##################################################
                    img_input = Variable(torch.cat(fixed_images + moving_images, dim=1).float()).to(self.device)

                    ### TODO: less hardcoded indices
                    if args.multichdata[3] == '1':
                        idx_t2w, idx_mov_t2w = [0, 10], [10]
                        idx_ch2, idx_mov_ch2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19], \
                                               [11, 12, 13, 14, 15, 16, 17, 18, 19]  # second channel - DTI
                    else:
                        idx_t2w, idx_mov_t2w = [0, 2], [2]
                        idx_ch2, idx_mov_ch2 = [1, 3], [3]   # second channel - FA

                    nmodalities = img_input.shape[1] // 2

                    # TODO: delete this it's just for test
                    if epoch == self.start_epoch and i == 0 and phase == 'valid' and img_input_once is None:
                        img_input_once = utils.cuda(Variable(torch.zeros_like(img_input) + img_input))

                    img_cond = utils.cuda(
                        F.interpolate(torch.cat(moving_images, dim=1),
                                      size=tuple([x // 8 for x in img_input.shape[2:]]),
                                      mode='trilinear', align_corners=False))

                    # TRAIN
                    ##################################################
                    if phase == 'train':
                        self.Attn.train()
                        self.Reg1.eval()
                        self.Reg2.eval()

                        # Forward pass through registration networks
                        ##################################################
                        with torch.no_grad():
                            # Get velocity field from  T2w model
                            _, _, _, vel1 = self.Reg1(img_input[:, idx_t2w, ...])

                            # Get velocity field from FA or DTI model
                            _, _, _, vel2 = self.Reg2(img_input[:, idx_ch2, ...])

                            vel_ = torch.cat([vel1, vel2], dim=1)

                        # Forward pass through attention network
                        ##################################################
                        alpha = self.Attn(vel_, img_cond)
                        vel = torch.mul(vel1, alpha[:,[0],...]) + torch.mul(vel2, alpha[:,[1],...])

                        # Do a train loop
                        ##################################################
                        (rec_loss, ddf_loss, total_loss), _, _ = \
                               self.train_validate_loop(args, img_input, vel, nmodalities, loss_weights)

                        metrics['rec_loss_train'].append(rec_loss.item())
                        metrics['ddf_loss_train'].append(ddf_loss.item())
                        metrics['total_loss_train'].append(total_loss.item())

                        # Update network
                        ###################################################
                        self.r_optimizer.zero_grad()
                        total_loss.backward()
                        self.r_optimizer.step()


                    # VALIDATE
                    #######################################################
                    else:
                        img_input = utils.cuda(Variable(torch.zeros_like(img_input_once) + img_input_once))

                        self.Attn.train()
                        self.Reg1.eval()
                        self.Reg2.eval()

                        # Forward pass through registration networks
                        ##################################################
                        with torch.no_grad():
                            # Get velocity field from  T2w model
                            _, _, _, vel1 = self.Reg1(img_input[:, idx_t2w, ...])

                            # Get velocity field from FA or DTI model
                            _, _, _, vel2 = self.Reg2(img_input[:, idx_ch2, ...])

                            vel_ = torch.cat([vel1, vel2], dim=1)

                            # Forward pass through attention network
                            ##################################################
                            alpha = self.Attn(vel_, img_cond)

                            vel = torch.mul(vel1, alpha[:,[0],...]) + torch.mul(vel2, alpha[:,[1],...])

                        # Do a validation loop
                        ##################################################
                        (rec_loss, ddf_loss, total_loss), \
                            (warpfull_mov, disp_field_pos, disp_field_neg), \
                            (warphalf_mov, warphalf_fix, disp_half_pos, disp_half_neg) = \
                               self.train_validate_loop(args, img_input, vel, nmodalities, loss_weights)

                        metrics['rec_loss_valid'].append(rec_loss.item())
                        metrics['ddf_loss_valid'].append(ddf_loss.item())
                        metrics['total_loss_valid'].append(total_loss.item())

                        # Save valid losses here:
                        rec_loss_valid += rec_loss.item()
                        ddf_loss_valid += ddf_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True

                            utils.plot_attention_maps(args, epoch, alpha)

                            utils.plot_mov_fix_moved(args, epoch,
                                               img_input[:, :nmodalities, ...],
                                               img_input[:, nmodalities:, ...],
                                               warpfull_mov,
                                               disp_field_pos, disp_field_neg,
                                               vol_size_=self.vol_size, device_=self.device, extra_title_='full')

                            if args.is_halfway:
                                utils.plot_mov_fix_moved(args, epoch,
                                                         img_input[:, :nmodalities, ...],
                                                         warphalf_mov, warphalf_fix,
                                                         disp_half_pos, disp_half_neg,
                                                         vol_size_=self.vol_size, device_=self.device,
                                                         extra_title_='half')

                        # Save best after all validation steps
                        #######################################################
                        if i >= (args.validation_steps - 1):
                            rec_loss_valid /= args.validation_steps
                            ddf_loss_valid /= args.validation_steps

                            print('AVG REC LOSS VALID | ', rec_loss_valid)
                            print('AVG DDF LOSS VALID | ', ddf_loss_valid)

                            # Save best
                            if best_reconstruction_loss > rec_loss_valid and epoch > 0:
                                best_reconstruction_loss = rec_loss_valid
                                print("Best Reconstruction Valid Loss %.2f" % (best_reconstruction_loss))

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Attn': self.Attn.state_dict(),
                                                       'r_optimizer': self.r_optimizer.state_dict(),
                                                       'losses_train': self.losses_train},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Rec Loss: %f' % (epoch + 1, rec_loss_valid))

                            # Stop early -- Don't go through all the validation set
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print(
                        "%s Epoch: (%3d) (%5d/%5d) (%3d) | Total Loss: %.2e | Rec Loss:%.2e | DDF Loss:%.2e | %.0fm %.2fs" %
                        (phase.upper(), epoch, i + 1, len_dataloader, step,
                         total_loss.item(), rec_loss.item(), ddf_loss.item(), time_elapsed // 60,
                         time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Attn': self.Attn.state_dict(),
                                   'r_optimizer': self.r_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.r_lr_scheduler.step()

        return self.losses_train


    def train_validate_loop(self, args_, img_input_, vel_, nmodalities_, loss_weights_):

        # Transform vel_ field into displacement field
        ##################################################
        flow_pos, flow_neg = vel_, -vel_

        # Scaling and squaring
        disp_field_pos_, disp_half_pos_ = self.integrate(flow_pos)
        disp_field_neg_, disp_half_neg_ = self.integrate(flow_neg)

        # Warp the moving all the way
        ##################################################
        warpfull_mov_ = self.transform(img_input_[:, nmodalities_:, ...], disp_field_pos_)

        # and the moving and fixed halfway (this is more time consuming)
        #############################
        if args_.is_halfway:
            warphalf_mov_ = self.transform(img_input_[:, nmodalities_:, ...], disp_half_pos_)
            warphalf_fix_ = self.transform(img_input_[:, :nmodalities_, ...], disp_half_neg_)

        # DTI is treated as a special case
        ##################################################
        if args_.multichdata[3] == '1':  # DTI is a channel

            # Calculate jacobian from the final displacement field directly
            ##################################################
            disp_field_pos_jac = units.calculate_jacobian(disp_field_pos_, self.vol_size, self.device)
            disp_field_pos_jac = disp_field_pos_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

            # Mask edges of jacobian
            ##################################################
            if self.dti_mask is not None:
                disp_field_pos_jac = (disp_field_pos_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)

            # Calculate rotation matrix from Jacobian doing polar decomposition
            ##################################################
            ## Note: Bare in mind that torch svd has the tendency of being extremely slow
            #        and I have empirically found that using the CPU version is much better
            u, _, vh = self.svd_layer(disp_field_pos_jac)
            R_pos = torch.matmul(u, vh)
            R_pos_tr = torch.transpose(R_pos, dim0=4, dim1=5)

            if args_.is_halfway:
                # Calculate jacobian from the halfway positive displacement
                ##################################################
                disp_field_half_pos_jac = units.calculate_jacobian(disp_half_pos_, self.vol_size, self.device)
                disp_field_half_pos_jac = disp_field_half_pos_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

                # Calculate jacobian from the halfway negative displacement
                ##################################################
                disp_field_half_neg_jac = units.calculate_jacobian(disp_half_neg_, self.vol_size, self.device)
                disp_field_half_neg_jac = disp_field_half_neg_jac.permute(0, 2, 3, 4, 1).reshape(1, *self.vol_size, 3, 3)

                # Mask edges of jacobian
                ##################################################
                if self.dti_mask is not None:
                    disp_field_half_pos_jac = (disp_field_half_pos_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)
                    disp_field_half_neg_jac = (disp_field_half_neg_jac * self.dti_mask) + (self.jac_ini * self.dti_mask_inv)

                # Calculate rotation matrix from Jacobian halfway pos doing polar decomposition
                ##################################################
                u, _, vh = self.svd_layer(disp_field_half_pos_jac)
                R_pos_half_pos = torch.matmul(u, vh)
                R_pos_tr_half_pos = torch.transpose(R_pos, dim0=4, dim1=5)

                # Calculate rotation matrix from Jacobian halfway neg doing polar decomposition
                ##################################################
                u, _, vh = self.svd_layer(disp_field_half_neg_jac)
                R_pos_half_neg = torch.matmul(u, vh)
                R_pos_tr_half_neg = torch.transpose(R_pos, dim0=4, dim1=5)

            # Reorient the tensors ONLY
            ##################################################
            # case 1 channel == DTI-only
            if len(loss_weights_) == 1:
                warpfull_mov_dti = torch.matmul(
                    torch.matmul(R_pos,
                                 warpfull_mov_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                             *self.vol_size,
                                                                             3, 3)),
                    R_pos_tr)

                if args_.is_halfway:
                    warphalf_mov_dti = torch.matmul(
                        torch.matmul(R_pos_half_pos,
                                     warphalf_mov_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                                 *self.vol_size,
                                                                                 3, 3)),
                        R_pos_tr_half_pos)

                    warphalf_fix_dti = torch.matmul(
                        torch.matmul(R_pos_half_neg,
                                     warphalf_fix_.permute(0, 2, 3, 4, 1).reshape(1,
                                                                                 *self.vol_size,
                                                                                 3, 3)),
                        R_pos_tr_half_neg)

                    warphalf_mov_dti = warphalf_mov_dti.view(1, *self.vol_size, 9)
                    warphalf_fix_dti = warphalf_fix_dti.view(1, *self.vol_size, 9)

            # case 2 channels == T2w + DTI (scalar + dti)
            else:  # len(loss_weights_) == 2:
                warpfull_mov_dti = torch.matmul(
                    torch.matmul(R_pos,
                                 warpfull_mov_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                         *self.vol_size,
                                                                                         3, 3)),
                    R_pos_tr)

                if args_.is_halfway:
                    warphalf_mov_dti = torch.matmul(
                        torch.matmul(R_pos_half_pos,
                                     warphalf_mov_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                             *self.vol_size,
                                                                                             3, 3)),
                        R_pos_tr_half_pos)

                    warphalf_fix_dti = torch.matmul(
                        torch.matmul(R_pos_half_neg,
                                     warphalf_fix_[:, 1:, ...].permute(0, 2, 3, 4, 1).reshape(1,
                                                                                             *self.vol_size,
                                                                                             3, 3)),
                        R_pos_tr_half_neg)

                    warphalf_mov_dti = warphalf_mov_dti.view(1, *self.vol_size, 9)
                    warphalf_fix_dti = warphalf_fix_dti.view(1, *self.vol_size, 9)

            warpfull_mov_dti = warpfull_mov_dti.view(1, *self.vol_size, 9)

            # Reconstruction Loss
            ###################################################
            rec_loss_ = 0

            # dti only
            if len(loss_weights_) == 1 and args_.multichdata[3] == '1':
                rec_loss_ += loss_weights_[0] * \
                            (self.RecLoss.loss(warpfull_mov_dti,
                                               img_input_[:, :nmodalities_, ...].permute(0, 2, 3, 4, 1)))
                if args_.is_halfway:
                    rec_loss_ += loss_weights_[0] * self.RecLoss.loss(warphalf_fix_dti, warphalf_mov_dti)

            # t2w + dti
            elif len(loss_weights_) == 2 and args_.multichdata[3] == '1':
                rec_loss_ += loss_weights_[0] * \
                            (self.RecLoss1.loss(warpfull_mov_[:, :1, ...],
                                                img_input_[:, :1, ...]))
                rec_loss_ += loss_weights_[1] * \
                            (self.RecLoss2.loss(warpfull_mov_dti,
                                                img_input_[:, 1:nmodalities_, ...].permute(0, 2, 3, 4, 1)))

                if args_.is_halfway:
                    rec_loss_ += loss_weights_[0] * \
                                self.RecLoss1.loss(warphalf_fix_[:, :1, ...], warphalf_mov_[:, :1, ...])

                    rec_loss_ += loss_weights_[1] * \
                                self.RecLoss2.loss(warphalf_fix_dti, warphalf_mov_dti)

        else:
            # Reconstruction Loss
            ###################################################
            rec_loss_ = 0

            for ilamda_, lamda_ in enumerate(loss_weights_):
                rec_loss_ += lamda_ * self.RecLoss.loss(img_input_[:, ilamda_:ilamda_ + 1, ...],
                                                       warpfull_mov_[:, ilamda_:ilamda_ + 1, ...])

                if args_.is_halfway:
                    rec_loss_ += lamda_ * self.RecLoss.loss(warphalf_fix_[:, ilamda_:ilamda_ + 1, ...],
                                                            warphalf_mov_[:, ilamda_:ilamda_ + 1, ...])

        # Smoothness Loss
        ###################################################
        ddf_loss_ = self.DDFLoss.loss(disp_field_pos_, loss_type='be') * args_.lamda_ddf

        # Total Loss
        ###################################################
        total_loss_ = args_.lamda_rec * (rec_loss_ + ddf_loss_)

        # Return losses and intermediate values
        ###################################################
        if args_.is_halfway:
            return (rec_loss_, ddf_loss_, total_loss_),\
                   (warpfull_mov_, disp_field_pos_, disp_field_neg_), \
                   (warphalf_mov_, warphalf_fix_, disp_half_pos_, disp_half_neg_)
        else:
            return (rec_loss_, ddf_loss_, total_loss_), \
                   (warpfull_mov_, disp_field_pos_, disp_field_neg_), \
                   (None, None, None, None)

