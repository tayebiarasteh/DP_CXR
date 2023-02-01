"""
Created on Jan 30, 2023.
age_sex_fairness.py

@author: Alexander Ziller <alex.ziller@tum.de>
"""

#%%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# %%
def calc_average_acc_other(
    sample_sizes: pd.Series, accuracies: pd.Series, label_key: str
) -> float:
    sample_sizes = sample_sizes.drop(label_key)
    accuracies = accuracies.drop(label_key)
    return np.average(accuracies, weights=sample_sizes)


def calc_accuracy_rate(sample_sizes: pd.Series, accuracies: pd.Series, label_key: str):
    return (
        calc_average_acc_other(sample_sizes, accuracies, label_key)
        / accuracies[label_key]
    )


def calc_sample_rates(sample_sizes: pd.Series, label_key: str):
    return ((sample_sizes.sum() - sample_sizes[label_key])) / sample_sizes[label_key]


def statistical_parity(sample_sizes: pd.Series, accuracies: pd.Series, label_key: str):
    return accuracies[label_key] - calc_average_acc_other(
        sample_sizes, accuracies, label_key
    )


def statistical_parity_binary(accuracies, priveleged_key, unpriveleged_key):
    return accuracies[unpriveleged_key] - accuracies[priveleged_key]


# %%
df = pd.read_csv("age_results_accuracy.csv", index_col="epsilon")
# %%
sample_size_idx = 0
model_idx = 1
key = "zero_thirty"
# %%
all_spds = {}
for model_idx in range(1, len(df)):
    sample_sizes = df.iloc[sample_size_idx]
    accuracies = df.iloc[model_idx]

    spds = {}
    for key in df.keys():
        accuracy_rate = calc_accuracy_rate(sample_sizes, accuracies, key)
        sample_rate = calc_sample_rates(sample_sizes, key)

        sp = statistical_parity(sample_sizes, accuracies, key)
        print(f"{key}:")
        print(f"\t Accuracy rate: {accuracy_rate:.4f}")
        print(f"\t Sample rate: {sample_rate:.4f}")
        print(f"\t SP: {sp:.2f}%")
        spds[key] = sp
    all_spds[df.index[model_idx]] = spds
new_df = pd.DataFrame(all_spds).T


# %%
new_df.to_csv("age_statistical_parity.csv")
# %%
df = pd.read_csv("sex_results_accuracy.csv", index_col="epsilon")
sps = []
for i in range(len(df)):
    sp = statistical_parity_binary(df.iloc[i], "Males", "Females")
    sps.append(sp)
df["sp"] = sps
df.to_csv("sex_statistical_parity.csv")
# %%
priv_df = df.iloc[1:]
pearsonr(priv_df.index, priv_df.sp)
