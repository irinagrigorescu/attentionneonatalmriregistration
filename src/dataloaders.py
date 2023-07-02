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
# dataloaders.py
#
##############################################################################

from __future__ import print_function, division
import os
import torch
import torchio
from torchvision.transforms import Compose
import numpy as np
import pandas as pd

from src.utils import MAX_DTI


def remove_nans(x):
    x[torch.isnan(x)] = 0
    return x


def remove_impossible_values(x):
    x[x > MAX_DTI] =  MAX_DTI
    x[x <-MAX_DTI] = -MAX_DTI
    x = x / MAX_DTI
    return x


class ImageRegistrationDataLoaderTorchio():
    """
    T2w data loader for image registration using TorchIO
    """

    def __init__(self, csv_file, root_dir, output_size, multichdata='1000',
                 is_augment=False, shuffle=False, is_dti=False, is_cgmfix=False, is_icfix=False,
                 is_internal_capsule_data=False):
        """
        Constructor
        :param csv_file: Path to the csv file with GA, AS, gender, filename
        :param root_dir: Path to data
        :param output_size: tuple (w,h,d)
        :param multichdata: string xxxx (0|1 t2w, 0|1 lab, 0|1 fa, 0|1 dti) e.g. 1110 return t2w, lab and fa
        :param is_augment: True | False
        :param shuffle: if True reshuffles indices for every get item
        """
        self.data_file = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.input_folder = root_dir
        self.output_size = output_size
        self.shuffle = shuffle
        self.is_dti = is_dti
        self.multichdata = multichdata
        self.indices = np.arange(len(self.data_file))   # indices of the data [0 ... N-1]
        self.is_augment = is_augment
        self.is_cgmfix = is_cgmfix
        self.is_icfix = is_icfix
        self.is_internal_capsule_data = is_internal_capsule_data

        self.subjects_list = self.get_subjects()
        self.transform = self.get_transform()

        self.subjects_dataset = torchio.SubjectsDataset(subjects=self.subjects_list,
                                                        transform=self.transform)


    def make_patches_dataloader(self, num_workers=4, queue_length=150, samples_per_volume=1, batch_size=1):
        queue_length = queue_length
        samples_per_volume = samples_per_volume

        sampler = torchio.data.UniformSampler(self.output_size)

        patches_queue = torchio.Queue(
            self.subjects_dataset,
            queue_length,
            samples_per_volume,
            sampler,
            num_workers=num_workers,
            shuffle_subjects=self.shuffle,
            shuffle_patches=False,
        )

        return patches_queue  # DataLoader(patches_queue, batch_size=batch_size)


    def get_transform(self):
        preprocessing = self.get_preprocessing()
        augmentation = self.get_augmentation()

        # Put them all together
        if self.is_augment:
            return Compose(preprocessing + augmentation)
        else:
            return Compose(preprocessing)


    def get_augmentation(self):
        to_motion = torchio.transforms.RandomMotion(degrees=2.0,
                                                    translation=2.0,  # 3.0
                                                    num_transforms=1,
                                                    p=0.75)
        to_spike = torchio.transforms.RandomSpike(num_spikes=1,
                                                  intensity=0.2,
                                                  p=0.75)

        # Choose to do motion, spike or both [NO AFFINE]
        augmentation_choice = np.random.choice([0, 1, 2])
        if augmentation_choice == 0:
            aug_img = [to_motion]
        elif augmentation_choice == 1:
            aug_img = [to_spike]
        else:
            aug_img = [to_motion, to_spike]

        return aug_img


    def get_preprocessing(self):
        # Remove NaNs
        nans_remove = torchio.transforms.Lambda(remove_nans)
        # Remove impossible dti values
        dti_remove = torchio.transforms.Lambda(remove_impossible_values,
                                               types_to_apply=['DTI', 'dti'])
        # Canonical reorientation and resampling
        to_ras = torchio.transforms.ToCanonical()
        # # Resampling to 0.75 isotropic - skip this, data already in that resolution
        to_iso = torchio.transforms.Resample((1.0, 1.0, 1.0))  #(0.75, 0.75, 0.75))  # 1.0,1.0,1.0
        # Z-Normalisation
        to_znorm = torchio.transforms.ZNormalization(exclude=['DTI', 'dti'])
        # Rescaling
        to_rescl = torchio.transforms.RescaleIntensity(out_min_max=(0.0, 1.0),
                                                       exclude=['DTI', 'dti'])
        # Crop or pad
        to_crop_pad = torchio.transforms.CropOrPad(target_shape=self.output_size,
                                                   padding_mode=0) #'minimum')

        # return [nans_remove, to_ras, to_iso, to_znorm, to_rescl, to_crop_pad]
        return [nans_remove, dti_remove, to_ras, to_iso, to_znorm, to_rescl, to_crop_pad]


    def get_subjects(self):
        subjects = []

        for item in self.indices:
            # Get t2w|lab|dti|fa names:
            t2w_name = os.path.join(self.input_folder,
                                    self.data_file.iloc[item, 0])
            lab_name = os.path.join(self.input_folder,
                                    self.data_file.iloc[item, 1])
            dti_name = os.path.join(self.input_folder,
                                    self.data_file.iloc[item, 2])
            fa_name = os.path.join(self.input_folder,
                                   self.data_file.iloc[item, 3])

            # To read or not to read internal capsule data
            if self.is_internal_capsule_data:
                try:
                    ic_name = os.path.join(self.input_folder,
                                           self.data_file.iloc[item, 0].split('_T2')[0] + '_ic_m.nii.gz')
                except:
                    print(f"Test subject {ic_name.split('/')[-1]} does not exist")
                    ic_name = None
            else:
                ic_name = None

            # Get separate cortical gray matter if it is for the fixed image
            if self.is_cgmfix:
                cgmfix_name = os.path.join(self.input_folder,
                                           self.data_file.iloc[item, 7])

                print(t2w_name, lab_name, dti_name, fa_name, cgmfix_name)

            # Get separate cortical gray matter if it is for the fixed image
            if self.is_icfix:
                icfix_name = os.path.join(self.input_folder,
                                           self.data_file.iloc[item, 8])

                print(t2w_name, lab_name, dti_name, fa_name, icfix_name)

            # Skip subject if file does not exist
            if self.multichdata[0] == '1' and not os.path.exists(t2w_name):
                print('t2w not exist')
                continue
            if self.multichdata[1] == '1' and not os.path.exists(lab_name):
                print('lab not exist')
                continue
            if self.multichdata[2] == '1' and not os.path.exists(fa_name):
                print('fa not exist')
                continue
            if self.multichdata[3] == '1' and not os.path.exists(dti_name):
                print('dti not exist')
                continue

            # Get ages, gender and names:
            ga_baby = float(self.data_file.iloc[item, 4])
            as_baby = float(self.data_file.iloc[item, 5])
            ge_baby = self.data_file.iloc[item, 6]

            if self.is_cgmfix:
                subj_name = self.data_file.iloc[item, 0].split('.nii')[0]
            else:
                subj_name = self.data_file.iloc[item, 0].split('_T2_')[0]

            # Check and load data
            subject_dict = {'GA': ga_baby,
                            'AS': as_baby,
                            'gender': ge_baby,
                            'name': subj_name}

            if self.multichdata[0] == '1':
                subject_dict['t2w'] = torchio.ScalarImage(t2w_name)
            if self.multichdata[1] == '1':
                subject_dict['lab'] = torchio.LabelMap(lab_name)
            if self.multichdata[2] == '1':
                subject_dict['fa'] = torchio.ScalarImage(fa_name)
            if self.multichdata[3] == '1':
                subject_dict['dti'] = torchio.Image(dti_name, type='DTI')
            if self.is_cgmfix:
                subject_dict['cgmfix'] = torchio.ScalarImage(cgmfix_name)
            if self.is_icfix:
                subject_dict['icfix'] = torchio.ScalarImage(icfix_name)
            if self.is_internal_capsule_data and ic_name is not None:
                subject_dict['ic'] = torchio.ScalarImage(ic_name)

            # Read subject
            subject = torchio.Subject(
                subject_dict
            )

            # Add to list
            subjects.append(subject)

        # print(len(subjects))

        return subjects
