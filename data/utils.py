"""
Created on Jan 3, 2023.
utils.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import pdb
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")




def statistics_creator():
    path = "master_list.csv"

    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[df['age'] > 0]
    df = df[df['age'] < 120]

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    plt.rcParams.update({'font.size': 22})

    plt.subplot(221)
    plt.hist(df_train['age'])
    plt.title('(A) Training set')

    plt.subplot(222)
    plt.hist(df_test['age'])
    plt.title('(B) Test set')

    plt.subplot(212)
    plt.hist(df['age'])
    plt.title('(C) Overall')
    plt.show()


def comorbidites_histogram():
    path = "UKA_master_list.csv"

    df = pd.read_csv(path, sep=',', low_memory=False)
    df = df[df['age'] >= 0]
    df = df[df['age'] < 120]
    # df = df[df['split'] == 'train']
    df = df[df['split'] == 'test']


    df_first = df[df['age'] >= 0]
    df_first = df_first[df_first['age'] < 30]
    mean = df_first['comorbidities'].mean()
    std = df_first['comorbidities'].std()
    print(f'[0 30]: {mean:.1f} ± {std:.1f}')

    df_sec = df[df['age'] >= 30]
    df_sec = df_sec[df_sec['age'] < 60]
    mean = df_sec['comorbidities'].mean()
    std = df_sec['comorbidities'].std()
    print(f'[30 60]: {mean:.1f} ± {std:.1f}')

    df_third = df[df['age'] >= 60]
    df_third = df_third[df_third['age'] < 70]
    mean = df_third['comorbidities'].mean()
    std = df_third['comorbidities'].std()
    print(f'[60 70]: {mean:.1f} ± {std:.1f}')

    df_fourth = df[df['age'] >= 70]
    df_fourth = df_fourth[df_fourth['age'] < 80]
    mean = df_fourth['comorbidities'].mean()
    std = df_fourth['comorbidities'].std()
    print(f'[70 80]: {mean:.1f} ± {std:.1f}')

    df_fifth = df[df['age'] >= 80]
    df_fifth = df_fifth[df_fifth['age'] < 100]
    mean = df_fifth['comorbidities'].mean()
    std = df_fifth['comorbidities'].std()
    print(f'[80 100]: {mean:.1f} ± {std:.1f}')

    df_female = df[df['gender'] > 0] # female
    mean = df_female['comorbidities'].mean()
    std = df_female['comorbidities'].std()
    print(f'female: {mean:.1f} ± {std:.1f}')

    df_male = df[df['gender'] < 1] # male
    mean = df_male['comorbidities'].mean()
    std = df_male['comorbidities'].std()
    print(f'male: {mean:.1f} ± {std:.1f}')

    mean = df['comorbidities'].mean()
    std = df['comorbidities'].std()
    print(f'overall: {mean:.1f} ± {std:.1f}')

    plt.rcParams.update({'font.size': 16})

    plt.suptitle('Distribution of comorbidities over the test set')

    plt.subplot(241)
    plt.hist(df_first['comorbidities'], bins=8)
    plt.title('(A) [0, 30) Years')
    plt.ylim([0, 6000])

    plt.subplot(242)
    plt.hist(df_sec['comorbidities'], bins=8)
    plt.title('(B) [30, 60) Years')
    plt.ylim([0, 6000])

    plt.subplot(243)
    plt.hist(df_third['comorbidities'], bins=8)
    plt.title('(C) [60, 70) Years')
    plt.ylim([0, 6000])

    plt.subplot(244)
    plt.hist(df_fourth['comorbidities'], bins=8)
    plt.title('(D) [70, 80) Years')
    plt.ylim([0, 6000])

    plt.subplot(245)
    plt.hist(df_fifth['comorbidities'], bins=8)
    plt.title('(E) [80, 100) Years')
    plt.ylim([0, 6000])

    plt.subplot(246)
    plt.hist(df_female['comorbidities'], bins=8, color='red')
    plt.title('(F) Female')
    plt.ylim([0, 10000])

    plt.subplot(247)
    plt.hist(df_male['comorbidities'], bins=8, color='red')
    plt.title('(G) Male')
    plt.ylim([0, 10000])

    plt.subplot(248)
    plt.hist(df['comorbidities'], bins=8,  color='green')
    plt.title('(H) Overall')
    plt.ylim([0, 16000])
    plt.show()



def sample_size():
    path = "UKA_master_list.csv"

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
    def __init__(self, cfg_path="DP_CXR/config/config.yaml"):
        pass


    def UKA(self):
        final_df = pd.DataFrame(columns=['image_id', 'split', 'subset' 'age', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'gender', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'atelectasis_right',	'atelectasis_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        label_path = 'master_list.csv'
        output_path = 'master_list.csv'
        df = pd.read_csv(label_path, sep=',')
        df = df[df['split'] == 'test']

        for index, row in tqdm(df.iterrows()):
            age = self.date_to_age(row['birth_date'], row['examination_date'])
            if row['patient_sex'] == 'M':
                gender = 0
            else:
                gender = 1

            tempp = pd.DataFrame([[row['image_id'], row['split'], row['subset'], age, row['birth_date'], row['examination_date'], row['study_time'],
                                            row['patient_sex'], gender, row['ExposureinuAs'], row['cardiomegaly'], row['congestion'], row['pleural_effusion_right'], row['pleural_effusion_left'],
                                   row['pneumonic_infiltrates_right'], row['pneumonic_infiltrates_left'], row['atelectasis_right'], row['atelectasis_left'], row['pneumothorax_right'], row['pneumothorax_left'], row['subject_id']]],
                                 columns=['image_id', 'split', 'subset', 'age', 'birth_date', 'examination_date', 'study_time',
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
    statistics_creator()

