# %%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('../../..')
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
adata = sc.read_h5ad(DATA_PATH + "PsychAD/c92_only_obs_obsm.h5ad")
# obs = adata.obs[adata.obs['c02_100v100_mask']]
obs = adata.obs

obs['PAC'] = PASCode.pac.assign_pac(
    scores = obs['pac_score'].values, # NOTE
    mode='cutoff', cutoff=0.5
)

#%%
################################################################################
################################################################################
#%% [markdown] 
### Evaluate on RF subject-level
from sklearn.ensemble import RandomForestClassifier
import shap

subid_col = 'SubID'
cond_col = 'c92x'
pos_cond = 'Depression_Mood'
neg_cond = 'Control'
class_col = 'subtype' # TODO

#
scmat = PASCode.utils.scmatrix(
    adata.obs, 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo = PASCode.utils.subject_info(
    adata.obs,
    subid_col,
    columns=[cond_col])
scmat = scmat.loc[subinfo.index]

scmat_trn = PASCode.utils.scmatrix(
    adata.obs[adata.obs['c92_100v100_mask']], 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo_trn = PASCode.utils.subject_info(
    adata.obs[adata.obs['c92_100v100_mask']],
    subid_col,
    columns=[cond_col])
scmat_trn = scmat_trn.loc[subinfo_trn.index]

#%%
### cell type prioritization by mean shap values across 100 runs
from sklearn.ensemble import RandomForestClassifier

num_repeat = 100

shap_values = np.zeros_like(scmat.values)
subinfo['subject_score'] = 0.0
for seed in np.arange(num_repeat):
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X=scmat_trn, 
            y=subinfo_trn[cond_col].map({pos_cond: 1, neg_cond: 0}))
    explainer = shap.TreeExplainer(clf)
    shap_values += explainer.shap_values(scmat.values)[1]

    subinfo['subject_score'] += clf.predict_proba(scmat)[:, 1]
subinfo['subject_score'] /= num_repeat
shap_values /= num_repeat

#%%
import matplotlib.colors as mcolors
# cmap = mcolors.LinearSegmentedColormap.from_list('custom', [(0, 'blue'), (1, 'red')])
cmap = mcolors.LinearSegmentedColormap.from_list('custom', [(0, '#9e471b'), (1, '#0e38c2')])

shap_values = pd.DataFrame(shap_values, index=scmat.index, columns=scmat.columns)
shap.summary_plot(
    shap_values.values, 
    scmat.values,  
    feature_names=scmat.columns,
    max_display=10,
    show=False,
    cmap=cmap,
    )

plt.savefig(
    './c92_subtype_prioritization_by_SHAP.pdf',  # TODO
    format='pdf', bbox_inches='tight', dpi=300)
plt.show()

shap_values.to_csv("./c92_shap_values_subtype.csv") # TODO

shap_top = shap_values.abs().mean(axis=0).sort_values(ascending=False).index[:10]

# %%
# TODO
shap_top = ['Astro', 'Oligo', 'OPC', 'IN_SST', 'Micro', 'IN_ADARB2',
                  'EN_L2_3_IT', 'IN_VIP', 'Endo', 'IN_PVALB_CHC']
shap_top = ['Astro_WIF1', 'Oligo_OPALIN', 'Astro_GRIA1', 'Astro_ADAMTSL3', 
            'IN_SST_EDNRA', 'EN_L2_3_IT_PDGFD', 'Micro', 'IN_ADARB2_RAB37', 
            'Oligo_RBFOX1', 'Astro_PLSCR1']


## PAC proportion by subclass
df0 = obs.groupby(['subtype', 'PAC']).size().unstack() # TODO
df = df0.div(df0.sum(axis=1), axis='index').copy()
df = df.drop(columns=[0])
df = df.rename(columns={-1: 'PAC-', 1: 'PAC+'})
df = df.sort_values(by='PAC+', ascending=False)
df0 = df0.loc[df.index]
df = df.loc[shap_top]

#%%
df.plot.bar(
    figsize=(10, 5),
    title='PAC proportion by subtype', # TODO
    ylabel='Cell number proportion',
    xlabel='',
    rot=90,
    color=['#9e471b',  '#0e38c2'],
    ylim=(0, 0.4),
)
plt.savefig('c92_PAC_subtype_proportion.pdf', bbox_inches='tight', dpi=300) # TODO
plt.show()

# %%
