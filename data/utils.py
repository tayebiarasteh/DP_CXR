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
from tqdm import tqdm
import os

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





class csv_summarizer():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/DP_CXR/config/config.yaml"):
        pass


    def UKA(self):
        final_df = pd.DataFrame(columns=['image_id', 'split', 'age', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'gender', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'atelectasis_right',	'atelectasis_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        label_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/DP_project_also_original/original_novalid_UKA_master_list.csv'
        output_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/DP_project_also_original/neworiginal_novalid_UKA_master_list.csv'
        df = pd.read_csv(label_path, sep=',')

        for index, row in tqdm(df.iterrows()):
            age = self.date_to_age(row['birth_date'], row['examination_date'])
            if row['patient_sex'] == 'M':
                gender = 0
            else:
                gender = 1

            tempp = pd.DataFrame([[row['image_id'], row['split'], age, row['birth_date'], row['examination_date'], row['study_time'],
                                            row['patient_sex'], gender, row['ExposureinuAs'], row['cardiomegaly'], row['congestion'], row['pleural_effusion_right'], row['pleural_effusion_left'],
                                   row['pneumonic_infiltrates_right'], row['pneumonic_infiltrates_left'], row['atelectasis_right'], row['atelectasis_left'], row['pneumothorax_right'], row['pneumothorax_left'], row['subject_id']]],
                                 columns=['image_id', 'split', 'age', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'gender', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left',	'atelectasis_right',	'atelectasis_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])
            final_df = final_df.append(tempp)
            final_df = final_df.sort_values(['image_id'])
            final_df.to_csv(output_path, sep=',', index=False)

        final_df.to_csv(output_path, sep=',', index=False)



    def date_to_age(self, birth_date, examination_date):
        """
        gives age in years
        """

        day_birth = int(os.path.basename(birth_date).split("-")[2])
        month_birth = int(os.path.basename(birth_date).split("-")[1])
        year_birth = int(os.path.basename(birth_date).split("-")[0])

        day_record = int(os.path.basename(examination_date).split("-")[2])
        month_record = int(os.path.basename(examination_date).split("-")[1])
        year_record = int(os.path.basename(examination_date).split("-")[0])

        diff_d = day_record - day_birth
        diff_m = month_record - month_birth
        diff_y = year_record - year_birth

        diff_d /= 360
        diff_m /= 12
        age = diff_y + diff_m + diff_d
        return age




if __name__ == '__main__':
    # sample_size()
    handle = csv_summarizer()
    handle.UKA()
