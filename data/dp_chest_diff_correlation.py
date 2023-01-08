"""
Created on Jan 8, 2023.
dp_chest_diff_correlation.py

@author: Alexander Ziller <alex.ziller@tum.de>
"""


import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sn

# %%
sn.set_theme(context="notebook", style="white", font="Times", font_scale=1.25, palette='viridis')
sn.despine()
colors = {"red": "firebrick", 
          "blue": "steelblue", 
          "green": "forestgreen",
          "purple": "darkorchid",
          "orange": "darkorange",
          "gray": "lightslategray",
          "black": "black"
         }
# %%
df = pd.read_csv('difference_correlation.csv')
df = df.astype({"perf_diff": float, "perf": float, "priv_perf":float, "num_samples": int})
df["priv_perf"] *= 100.0
# %%
r, p = pearsonr(df.perf_diff.to_numpy(), df.perf.to_numpy())
print(f"Performance diff has r-value of {r:.2f} with p-value of {p:.2f}")
# %%
r, p = pearsonr(df.perf_diff.to_numpy(), df.num_samples.to_numpy())
print(f"Num samples has r-value of {r:.2f} with p-value of {p:.2f}")

# %%
r, p = pearsonr(df.perf.to_numpy(), df.num_samples.to_numpy())
print(f"Non-private Performance has r-value of {r:.2f} with p-value of {p:.2f}")
r, p = pearsonr(df.priv_perf.to_numpy(), df.num_samples.to_numpy())
print(f"Private Performance has r-value of {r:.2f} with p-value of {p:.2f}")

# %%
plt.scatter(df.num_samples, df.priv_perf, c='magenta', label='Private', marker='o')
plt.scatter(df.num_samples, df.perf, c='orange', label='Non-private', marker='o')
for i, (num_samples, perf, priv_perf) in enumerate(zip(df.num_samples, df.perf, df.priv_perf)):
    plt.plot((num_samples, num_samples), (perf, priv_perf), marker='|', c='black', alpha=0.8)
    plt.annotate(df.name.iloc[i], (num_samples, perf - ((perf-priv_perf)/2)), fontsize=6, ha='center')
    # plt.annotate(df.name.iloc[i], (22000, perf), fontsize=6, rotation=0)
# plt.quiver(df.num_samples, df.perf, 0, -df.priv_perf, units='y', headaxislength=0, headlength=0, headwidth=0)
# plt.xlim(left=0, right=8e3)
plt.ylim((73, 100))
# plt.xlim(left=0)
plt.xscale('log')
plt.ylabel('AUC-ROC of all classes')
plt.xlabel('Sample size')
plt.legend()
plt.gca().yaxis.set_major_formatter('{x:.0f}%')
plt.savefig("samplesize_vs_performance.png", dpi=1200, bbox_inches='tight')

