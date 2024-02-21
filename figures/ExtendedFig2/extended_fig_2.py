# %%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('../..')
import PASCode

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PASCode.random_seed.set_seed(0)

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

#%% [markdown]
# adata = sc.read_h5ad(DATA_PATH + "PsychAD/c02.h5ad")
adata = sc.read_h5ad(DATA_PATH + "PsychAD/c02_only_obs_obsm.h5ad")
obs = adata.obs[adata.obs['c02_100v100_mask']]
obs['PAC'] = PASCode.pac.assign_pac(
    scores = obs['pac_score'].values, # NOTE
    mode='cutoff', cutoff=0.5
)

#%%
###############################################################################
# remove 19k TODO
###############################################################################
tormv = pd.read_csv("/home/che82/athan/PASCode/240124_PsychAD_freeze3_outlier_nuclei.csv", index_col=0)
adata = adata[~adata.obs.index.isin(tormv.index)]

#%% [markdown]
## PAC proportion by subclass
df0 = obs.groupby(['subclass', 'PAC']).size().unstack()
df = df0.div(df0.sum(axis=1), axis='index').copy()
df = df.drop(columns=[0])
df = df.rename(columns={-1: 'PAC-', 1: 'PAC+'})
df = df.sort_values(by='PAC+', ascending=False)
df0 = df0.loc[df.index]

#%% filter
# shap top 10 & ... TODO
shap_top = ['Micro', 'Astro', 'EN_L3_5_IT_2', 'EN_L2_3_IT', 'VLMC', 'Oligo', 'IN_SST', 'IN_PVALB', 'IN_VIP', 'EN_L3_5_IT_3']
df = df.loc[shap_top]
cnum=df0.sum(axis=1)

#%%
import seaborn as sns
df_melt = df.copy()
df_melt['subclass'] = list(df_melt.index.values)
df_melt = df_melt.melt(
    id_vars=['subclass'],
    value_vars=['PAC+', 'PAC-'],
    value_name="proportion")

plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='subclass', y='proportion', hue='PAC', data=df_melt, palette=['#591496',  '#1f7a0f'])
ax1 = plt.gca()
ax2 = ax1.twinx() # create a secondary y-axis for the cell numbers
n_subclasses=  10
x_coords = range(n_subclasses)
ax2.plot(x_coords, cnum.loc[df.index], color='#cc7333', 
         linestyle='-', linewidth=3, marker='o', markersize=12)

ax1.set_ylabel('PAC proportion', fontsize=15)
ax2.set_ylabel('Cell numbers')
plt.title('PAC proportion by subclass using c02 100v100', fontsize=15)
ax1.legend(loc='upper right', fontsize=15)

ax1.set_ylim(0, 0.4)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right', fontsize=15)
ax1.set_xlabel('')

plt.savefig('PAC_subclass_proportion.pdf', bbox_inches='tight', dpi=100)
plt.show()

#%% [markdown]
## scmatrix swarmpot
scmat = PASCode.utils.scmatrix(
    obs,
    subid_col = 'SubID',
    class_col = 'subclass',
    score_col = 'pac_score',
)
subj_info = PASCode.utils.subject_info(
    obs, 
    subid_col='SubID',
    columns=['c02x', 'Sex']
)
assert all(scmat.index == subj_info.index)

scmat = scmat.loc[:, shap_top]

#%%
df_melted = scmat.melt(var_name="Celltype", value_name="Score")
df_melted['c02x'] = list(subj_info['c02x'].values) * scmat.shape[1]
#%%
_, ax = plt.subplots(figsize=(6, 12))
sns.swarmplot(
    x="Score", 
    y="Celltype", 
    hue='c02x',
    palette={'AD':'#591496', 'Control':'#1f7a0f'},
    data=df_melted,
    ax=ax,
    dodge=False,
    linewidth=0,
)

ax.set_xlim(-1, 1)
ax.axvline(x=0, color='black', linestyle='--', linewidth=3)
ax.axvline(x=-.25, color='blue', linestyle='--', linewidth=2)
ax.axvline(x=0.25, color='red', linestyle='--', linewidth=2)

plt.show()

# %%
