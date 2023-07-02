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
# train_attention.py
#
##############################################################################
from src.utils import ArgumentsTrainInferenceRegistration, plot_losses_train
from src import models as md


# ==================================================================================================================== #
#
#  TRAIN the multi-channel attention image registration
#
# ==================================================================================================================== #

N_epochs = 26

# Prepare arguments
############################################################

# Lambdas as weights to different losses
########################################
l_t2w = [1, 1]      # t2w
l_fa  = [1, 0]      # fa
l_dti = [0, 1]      # dti

l_lab = [0] * len(l_t2w)     # lab
l_kld = [1] * len(l_t2w)     # kld
l_ddf = [0.01] * len(l_t2w)  # ddf

l_rec = [5000] * len(l_t2w)

zsz = [ 32] * len(l_t2w)     # z latent space size
ssz = [0.9] * len(l_t2w)     # smoothing size

loss_type = ['NCC', 'NCCEDS']


for i in range(len(l_t2w)):

    exp_current = 'attn_'
    multichdata = ''
    calcchdata = ''

    for _, (ttype_name, ttype) in enumerate(zip(['t2w', 'lab', 'fa', 'dti'],
                                                [l_t2w, l_lab, l_fa, l_dti])):
        if ttype[i] != 0:
            exp_current += ttype_name + '_'
            multichdata += '1'

            if ttype_name != 'dti':
                calcchdata += '1'
            else:
                calcchdata += '1' * 9

        else:
            multichdata += '0'
            calcchdata += '0'

    exp = exp_current + 'sm' + str(ssz[i]) + '_zs' + str(zsz[i]) + '_' + loss_type[i] + '/'
    input_channels = sum([1 for x in calcchdata if x == '1']) * 2

    # Pre-trained reg1 is the T2w data
    checkpoint_reg1 = '/path/to/T2w/pretrained/checkpoints/folder/'

    # Pre-trained reg2 is the FA data or DTI data
    if multichdata[3] == '1':
        checkpoint_reg2 = '/path/to/DTI/pretrained/checkpoints/folder/'
    else:
        checkpoint_reg2 = '/path/to/FA/pretrained/checkpoints/folder/'

    input_nc_reg1 = 2
    input_nc_reg2 = input_channels - input_nc_reg1

    print(i, exp, input_channels)
    print(multichdata, exp_current[:-1])
    print('3D_' + exp)
    print('\n')

    # # # Prepare arguments
    args = ArgumentsTrainInferenceRegistration(n_features=[zsz[i]],
                                               epochs=N_epochs,
                                               decay_epoch=1,
                                               lr=0.001,
                                               gpu_ids=0,
                                               crop_height=128,
                                               crop_width=128,
                                               crop_depth=128,
                                               lamda_t2w=l_t2w[i],  # t2w
                                               lamda_fa=l_fa[i],    # fa
                                               lamda_dti=l_dti[i],  # dti
                                               lamda_lab=l_lab[i],  # lab
                                               lamda_kld=l_kld[i],  # kld
                                               lamda_ddf=l_ddf[i],  # ddf
                                               lamda_rec=l_rec[i],  # how rec weights against kld
                                               smooth=ssz[i],
                                               loss_type=loss_type[i],  # NCC
                                               validation_steps=4,
                                               training=True,
                                               multichdata=multichdata,
                                               root_dir_dHCP='/path/to/your/data/',
                                               csv_dir='/path/to/your/csv/file/',
                                               train_csv='data_train.csv',
                                               valid_csv='data_validate.csv',
                                               test_csv='data_test.csv',
                                               fixed_csv='data_train_fix.csv',
                                               results_dir='/path/to/the/folder/results-' + exp,
                                               checkpoint_dir='/path/to/the/folder/checkpoints-' + exp,                                              fixed_csv='dhcp_data_train_fix.csv',
                                               exp_name=exp_current[:-1],
                                               reg_net1='baseline3D',
                                               reg_net2='baseline3D',
                                               attn_net='attnvel3D',
                                               input_nc=6,
                                               output_nc=2,
                                               input_nc_reg1=input_nc_reg1,   # t2w
                                               input_nc_reg2=input_nc_reg2,   # fa or dti
                                               output_nc_reg=3,
                                               checkpoint_dir_reg1=checkpoint_reg1,
                                               checkpoint_dir_reg2=checkpoint_reg2,
                                               is_halfway=True)

    args.gpu_ids = [0]

    if args.training:
        print("Training")
        model = md.AttentionVelFieldImageRegistration(args)

        # Run train
        ###################
        losses_train = model.train(args)

        # Plot losses
        ###################
        plot_losses_train(args, losses_train, 'fig_losses_train_E')