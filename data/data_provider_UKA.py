"""
Created on Aug 6, 2022.
data_provider_UKA.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15




class UKA_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size256=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA/chest_radiograph')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.mode = mode

        if size256:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed256')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed')

        self.file_path_list = list(self.subset_df['image_id'])
        self.chosen_labels = ['cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left', 'pneumonic_infiltrates_right',
                              'pneumonic_infiltrates_left', 'atelectasis_right', 'atelectasis_left'] # 8 labels



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]['subset'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, subset, str(self.file_path_list[idx]) + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]

        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            if self.chosen_labels[idx] == 'cardiomegaly':
                if int(label_df[self.chosen_labels[idx]].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 4:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 1:
                    label[idx] = 0
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 2:
                    label[idx] = 0
            else:
                if int(label_df[self.chosen_labels[idx]].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

        label = label.float()

        if self.mode == 'test':
            gender = label_df['gender'].values[0]
            age = label_df['age'].values[0]

            return img, label, gender, age

        else:
            return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 3)
            disease_length += sum(train_df[diseases].values == 4)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class mimic_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size256=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.size256 = size256
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, "MIMIC")
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        self.subset_df = PAview.append(APview)
        self.file_path_list = list(self.subset_df['jpg_rel_path'])

        self.chosen_labels = ['cardiomegaly', 'consolidation', 'pleural_effusion', 'pleural_effusion', 'pneumonia', 'pneumonia', 'atelectasis', 'atelectasis'] # pretraining



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        if self.size256:
            img_path = img_path.replace("/files/", "/preprocessed256/")
        else:
            img_path = img_path.replace("/files/", "/preprocessed/")

        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor

