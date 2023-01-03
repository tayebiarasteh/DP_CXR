"""
Created on Jan 3, 2023.
utils.py
heatmap figures generator

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import pdb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")



def sample_size():
    path = "/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/DP_project_also_original/original_novalid_UKA_master_list.csv"

    df = pd.read_csv(path, sep=',', low_memory=False)

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    labellist = ['cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                 'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'atelectasis_right', 'atelectasis_left']

    for label in labellist:
        subsetdf1 = df_train[df_train[label] == 3]
        subsetdf2 = df_train[df_train[label] == 4]
        subsetdf_train = subsetdf1.append(subsetdf2)

        subsetdf1 = df_test[df_test[label] == 3]
        subsetdf2 = df_test[df_test[label] == 4]
        subsetdf_test = subsetdf1.append(subsetdf2)

        print(f'\n{label}: train: {len(subsetdf_train)} ({len(subsetdf_train) / len(df_train) * 100:.2f}%) | test: {len(subsetdf_test)} ({len(subsetdf_test) / len(df_test) * 100:.2f}%)')




if __name__ == '__main__':
    sample_size()

