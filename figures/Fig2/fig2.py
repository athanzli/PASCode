# %%
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.append('../..')
import PASCode

from PASCode.utils import plot_pac_umap, plot_umap, plot_legend

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PASCode.random_seed.set_seed(0)

DATA_PATH = "/home/che82/athan/ProjectPASCode/data/"

#%%
###############################################################################
# load data
###############################################################################
adata1 = sc.read_h5ad(DATA_PATH + 'PsychAD/c02_only_obs_obsm.h5ad')
adata2 = sc.read_h5ad(DATA_PATH + 'SEA-AD/SEAAD.h5ad')
adata3 = sc.read_h5ad(DATA_PATH + 'ROSMAP/ROSMAP_psychad_genes.h5ad')

class_palette_df = pd.read_csv('../class_palette.csv', index_col=0)
subclass_palette_df = pd.read_csv('../subclass_palette.csv', index_col=0)

class_palette = class_palette_df.to_dict()['color_hex']
subclass_palette = subclass_palette_df.to_dict()['color_hex']

# arange the order of subclasses in subclass_palette
classes = ['Astro', 'Immune', 'Oligo', 'OPC', 'EN', 'IN', 'Endo', 'Mural']
subclasses = ['Astro', 'Micro', 'Immune', 'PVM', 'Oligo', 'OPC', 'EN_L2_3_IT',
 'EN_L3_5_IT_1', 'EN_L3_5_IT_2', 'EN_L3_5_IT_3', 'EN_L5_6_NP',
 'EN_L5_ET', 'EN_L6B', 'EN_L6_CT', 'EN_L6_IT_1', 'EN_L6_IT_2',
  # 'EN_NF',  # NOTE removed!
 'IN_ADARB2', 'IN_LAMP5_LHX6', 'IN_LAMP5_RELN', 'IN_PVALB',
 'IN_PVALB_CHC', 'IN_SST', 'IN_VIP', 'Endo', 'PC', 'SMC', 'VLMC']
class_palette = {k: class_palette[k] for k in classes}
subclass_palette = {k: subclass_palette[k] for k in subclasses}

#%%
###############################################################################
# remove outliers for PsychAD data
###############################################################################
tormv = pd.read_csv("/home/che82/athan/ProjectPASCode/data/PsychAD/240124_PsychAD_freeze3_outlier_nuclei.csv", index_col=0)
adata1 = adata1[~adata1.obs.index.isin(tormv.index)]

#%%
###############################################################################
# create palette for original cell types of SEA-AD and ROSMAP
###############################################################################
subclass2_palette = {
    'Astrocyte': '#C19A6B',
    'Microglia-PVM': '#C11B17',
    'Oligodendrocyte': '#ECE5B6',
    'OPC': '#FFF380',
    'L2/3 IT': '#659EC7',
    'L4 IT': '#95B9C7',
    'L5 IT': '#6495ED',
    'L5/6 NP': '#0020C2',
    'L5 ET': '#4863A0',
    'L6b': '#488AC7',
    'L6 CT': '#3BB9FF',
    'L6 IT': '#B4CFEC',
    'L6 IT Car3': '#1589FF',
    'Lamp5 Lhx6': '#7BCCB5',
    'Lamp5': '#3B9C9C',
    'Pvalb': '#B2C248',
    'Sst': '#728C00',
    'Sst Chodl': '#728C00',
    'Vip': '#89C35C',
    'Chandelier': '#4E8975',
    'Endothelial': '#FFA62F',
    'Pax6': '#E0B0FF',
    'Sncg': '#89C35C',
    'VLMC': '#B93B8F'
}

class3_palette = {
    'Ast': '#C19A6B',
    'Mic': '#F75D59',
    'Oli': '#ECE5B6',
    'Opc': '#FFF380',
    'Ex': '#659EC7',
    'In': '#4E8975',
    'End': '#FFA62F',
    'Per': '#A52A2A',
}

#%%
###############################################################################
# plot umap
###############################################################################
#%% c02 subclass
handles, labels = handles, labels = plot_umap(
    adata1[adata1.obs['c02_100v100_mask']],
    umap_key='X_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./c02_trn_subclass_umap.tiff'
)
plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./psychAD_subclass_legend.pdf')

#%% [markdown] 
## c02 phenotype
phenotype_palette ={'AD': '#591496', 'Control': '#1f7a0f'}
handles, labels = handles, labels = plot_umap(
    adata1[adata1.obs['c02_100v100_mask']],
    umap_key='X_umap',
    class_col='c02x', 
    class_palette=phenotype_palette,
    text_on_plot=False,
    save_path='./c02_trn_phenotype_umap.tiff'
)

plot_legend(
     legend_handles=handles,
    legend_labels=labels,
    legend_ncol=1,
    save_path='./c02_phenotype_legend.pdf'
)

#%% c02 train (100v100) latent space subclass and PAC umap
handles, labels = plot_umap(
    adata1[adata1.obs['c02_100v100_mask']],
    umap_key='model_layer2_umap',
    class_col='subclass',
    class_palette=subclass_palette, 
    text_on_plot=True,
    save_path='./c02_train_latent_subclass.tiff'
)
plot_pac_umap(
    adata1[adata1.obs['c02_100v100_mask']], 
    umap_key='model_layer2_umap', 
    pac_col='pac_score',
    save_path='./c02_train_latent_PAC_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496']
)

#%% c02 test (11v11) latent space subclass and PAC umap
s = 8
mask = adata1.obs['c02_11v11_mask'].values

handles, labels = plot_umap(
    adata1[adata1.obs['c02_11v11_mask']],
    umap_key='X_umap',
    class_col='subclass',
    class_palette=subclass_palette, 
    text_on_plot=True,
    save_path='./c02_test_subclass_umap.tiff',
    s=s
)
plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='X_umap', 
    pac_col='pac_score',
    save_path='./c02_test_PAC_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
    s=s
)
plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='X_umap', 
    pac_col='rra_pac',
    save_path='./c02_test_AggLabel_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
    s=s
)

#%% SEAAD subclass umap
# handles, labels = plot_umap(
#     adata2,
#     umap_key='X_umap',
#     class_col='subclass', 
#     class_palette=subclass_palette, 
#     save_path='./SEAAD_subclass_umap.tiff'
# )
# plot_legend(
#     legend_handles=handles,
#     legend_labels=labels,
#     legend_ncol=2,
#     save_path='./PsychAD_subclass_legend.pdf'
# )

handles, labels = plot_umap(
    adata2,
    umap_key='X_umap',
    class_col='Subclass', 
    class_palette=subclass2_palette, 
    save_path='./SEAAD_OriginalSubclass_umap.tiff'
)
plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./SEAAD_OriginalSubclass_legend.pdf'
)

plot_pac_umap(
    adata2, 
    umap_key='X_umap',
    pac_col='pac_score',
    save_path='./SEAAD_PAC_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
)
plot_pac_umap(
    adata2, 
    umap_key='X_umap',
    pac_col='rra_pac',
    save_path='./SEAAD_AggLabel_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
)

#%% ROSMAP subclass umap
s=16

# handles, labels = plot_umap(
#     adata3,
#     umap_key='X_umap',
#     class_col='subclass', 
#     class_palette=subclass_palette, 
#     save_path='./ROSMAP_subclass_umap.tiff',
#     s=s
# )
# plot_legend(
#      legend_handles=handles,
#     legend_labels=labels,
#     legend_ncol=2,
#     save_path='./PsychAD_subclass_legend.pdf'
# )

handles, labels = plot_umap(
    adata3,
    umap_key='X_umap',
    class_col= 'broad.cell.type', 
    class_palette=class3_palette, 
    save_path='./ROSMAP_OriginalSubclass_umap.tiff',
    s=s
)
plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./ROSMAP_OriginalSubclass_legend.pdf'
)

plot_pac_umap(
    adata3, 
    umap_key='X_umap', 
    pac_col='pac_score',
    save_path='./ROSMAP_PAC_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
    s=s
)
plot_pac_umap(
    adata3, 
    umap_key='X_umap', 
    pac_col='rra_pac',
    save_path='./ROSMAP_AggLabel_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496'],
    s=s
)

#%%
###############################################################################
# average PAC score
###############################################################################
# #%% c02 left
# adata1_pac = sc.read_h5ad(DATA_PATH + 'PsychAD/c02_11v11.h5ad')

# adata1_pac.obs['rra_pac'] = PASCode.pac.assign_pac(
#     scores = adata1_pac.obs['rra_milo_meld_daseq'].values,
#     mode='cutoff', cutoff=0.5)
# print(adata1_pac.obs['rra_pac'].value_counts())
# adata1.obs.loc[adata1_pac.obs.index, 'rra_pac'] = adata1_pac.obs['rra_pac'].values
# adata1.obs['c02_11v11_mask'] = adata1.obs.index.isin(adata1_pac.obs.index)

# #%% seaad
# adata2.obs['rra_pac'] = PASCode.pac.assign_pac(
#     scores = adata2.obs['rra_milo_meld_daseq'].values,
#     mode='cutoff', cutoff=0.5)
# print(adata2.obs['rra_pac'].value_counts())
# adata2.obs['full_mask'] = np.full(adata2.shape[0], True)
# adata2.write_h5ad(DATA_PATH + 'SEA-AD/seaad.h5ad')

# #%% mit
# adata3_pac = sc.read_h5ad(DATA_PATH + 'ROSMAP/rosmap.h5ad')

# adata3_pac.obs['rra_pac'] = PASCode.pac.assign_pac(
#     scores = adata3_pac.obs['rra_milo_meld_daseq'].values,
#     mode='cutoff', cutoff=0.5)
# print(adata3_pac.obs['rra_pac'].value_counts())
# adata3.obs.loc[adata3_pac.obs.index, 'rra_pac'] = adata3_pac.obs['rra_pac'].values

#%% [markdown]
### Evaluate PAC-level prediction
# data_paths = [DATA_PATH + 'PsychAD/c02_only_obs_obsm.h5ad', 
#               DATA_PATH + 'SEA-AD/seaad.h5ad', 
#               DATA_PATH + 'ROSMAP/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad']

adatas = [adata1, adata2, adata3]
mask_names = ['c02_11v11_mask', 'full_mask', 'full_mask']
grp_names = ['c02x', 'SEA-AD', 'ROSMAP']
df = pd.DataFrame(columns=['Dataset', 'Aggr. label', 'Predicted PAC score'])

for i in range(3):
    adata = adatas[i]
    mask = adata.obs[mask_names[i]]
    y_true = adata.obs['rra_pac'][mask].map({-1:'Neg. aggr. label', 0:'Non-label', 1:'Pos. aggr. label'}).values
    y_pred = adata.obs['pac_score'][mask].values
    df = pd.concat([df, pd.DataFrame({
        'Dataset': grp_names[i],
        'Aggr. label': y_true,
        'Predicted PAC score': y_pred})])

#%% statistical test for pairs of groups
import scipy
p_values = []
for dataset in df['Dataset'].unique():
    data_current_dataset = df[df['Dataset'] == dataset]

    group1_scores = data_current_dataset[data_current_dataset['Aggr. label'] == 'Neg. aggr. label']['Predicted PAC score']
    group2_scores = data_current_dataset[data_current_dataset['Aggr. label'] == 'Non-label']['Predicted PAC score']
    group3_scores = data_current_dataset[data_current_dataset['Aggr. label'] == 'Pos. aggr. label']['Predicted PAC score']

    _, p_value_12 = scipy.stats.mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
    _, p_value_23 = scipy.stats.mannwhitneyu(group2_scores, group3_scores, alternative='two-sided')
    
    p_values.append((dataset, 'Group1-Group2', p_value_12))
    p_values.append((dataset, 'Group2-Group3', p_value_23))

print(p_values)

#%%
ordered_palette = ['#1f7a0f', 'white', '#591496']
plt.figure(figsize=(25, 30))
ax = sns.boxplot(
    data=df,
    x='Dataset',
    y='Predicted PAC score',
    hue='Aggr. label',
    hue_order=['Neg. aggr. label', 'Non-label', 'Pos. aggr. label'],
    palette=ordered_palette,
)
ax.set_xlabel('') 
ax.set_ylabel('Predicted PAC score', fontsize=80)
ax.tick_params(labelsize=75)
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=df['Aggr. label'].nunique(), 
    title=None,
    fontsize=60,
    title_fontsize=40,
)

plt.savefig('./predicted_pac_score_boxplot.pdf', format='pdf', bbox_inches='tight', dpi=100)
plt.show()

#%% [markdown]
### c02 100v100 scmatrix heatmap TODO use R
# adata = sc.read_h5ad(DATA_PATH + 'PsychAD/c02.h5ad')
# obs = adata.obs[adata.obs['c02_100v100_mask']]
# scmat = PASCode.utils.scmatrix(
#     obs,
#     'SubID',
#     class_col='subclass',
#     score_col='pac_score')

# _, ax = plt.subplots(figsize=(9, 18))
# cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#73AE65", "#8C72A6"])
# sns.heatmap(scmat.T, ax=ax, cmap=cmap, vmin=-1, vmax=1)
# plt.show()

#%% [markdown] 
############################################## Evaluate on RF subject-level
from sklearn.ensemble import RandomForestClassifier
import shap

subid_col = 'SubID'
cond_col = 'c02x'
pos_cond = 'AD'
neg_cond = 'Control'
class_col = 'subclass'
braak_col = 'r01x'
#
scmat = PASCode.utils.scmatrix(
    adata1.obs, 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo = PASCode.utils.subject_info(
    adata1.obs,
    subid_col,
    columns=[cond_col])
scmat = scmat.loc[subinfo.index]

scmat_trn = PASCode.utils.scmatrix(
    adata1.obs[adata1.obs['c02_100v100_mask']], 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo_trn = PASCode.utils.subject_info(
    adata1.obs[adata1.obs['c02_100v100_mask']],
    subid_col,
    columns=[cond_col])
scmat_trn = scmat_trn.loc[subinfo_trn.index]

## for R heatmap plot
scmat_trn_tmp = scmat_trn.loc[:, subclasses]
scmat_trn_tmp = scmat_trn_tmp.rename(columns={k: (i+1) for i, k in enumerate(subclasses)})
scmat_trn_tmp.to_csv("./c02_trn_scmat.csv")
subinfo_trn['c02x'].to_csv("./c02_trn_phenotype.csv")

scmat_tst1 = PASCode.utils.scmatrix(
    adata1.obs[~adata1.obs['c02_100v100_mask']],
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo_tst1 = PASCode.utils.subject_info(
    adata1.obs[~adata1.obs['c02_100v100_mask']],
    subid_col,
    columns=[cond_col, braak_col])
scmat_tst1 = scmat_tst1.loc[subinfo_tst1.index]

#
subid_col = 'Donor ID'
cond_col = 'Cognitive Status'
pos_cond = 'Dementia'
neg_cond = 'No dementia'
class_col = 'subclass'
braak_col = 'Braak'
scmat_tst2 = PASCode.utils.scmatrix(
    adata2.obs, 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo_tst2 = PASCode.utils.subject_info(
    adata2.obs,
    subid_col,
    columns=[cond_col, braak_col])
scmat_tst2 = scmat_tst2.loc[subinfo_tst2.index]

#
cond_col = 'diagnosis'
pos_cond = 'AD'
neg_cond = 'CTL'
subid_col = 'individualID'
braak_col = 'braaksc'
scmat_tst3 = PASCode.utils.scmatrix(
    adata3.obs, 
    subid_col,
    class_col=class_col,
    score_col='pac_score')
subinfo_tst3 = PASCode.utils.subject_info(
    adata3.obs,
    subid_col,
    columns=[cond_col, braak_col])
scmat_tst3 = scmat_tst3.loc[subinfo_tst3.index]

assert scmat.columns.equals(scmat_trn.columns)   & \
       scmat.columns.equals(scmat_tst1.columns) & \
       scmat.columns.equals(scmat_tst2.columns) & \
       scmat.columns.equals(scmat_tst3.columns)

# scmat = scmat.drop(columns=['EN_NF'])
# scmat_trn = scmat_trn.drop(columns=['EN_NF'])
# scmat_tst1 = scmat_tst1.drop(columns=['EN_NF'])
# scmat_tst2 = scmat_tst2.drop(columns=['EN_NF'])
# scmat_tst3 = scmat_tst3.drop(columns=['EN_NF'])

#%% [markdown]
### cell type prioritization by mean shap values across 100 runs
num_repeat = 100

cond_col = 'c02x'
pos_cond = 'AD'
neg_cond = 'Control'
shap_values1 = np.zeros_like(scmat.values)
shap_values2 = np.zeros_like(scmat_tst2.values)
shap_values3 = np.zeros_like(scmat_tst3.values)
subinfo['subject_score'] = 0.0
subinfo_tst1['subject_score'] = 0.0
subinfo_tst2['subject_score'] = 0.0
subinfo_tst3['subject_score'] = 0.0
for seed in np.arange(num_repeat):
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X=scmat_trn, 
            y=subinfo_trn[cond_col].map({pos_cond: 1, neg_cond: 0}))
    explainer = shap.TreeExplainer(clf)
    
    shap_values1 += explainer.shap_values(scmat.values)[1]
    shap_values2 += explainer.shap_values(scmat_tst2.values)[1]
    shap_values3 += explainer.shap_values(scmat_tst3.values)[1]

    subinfo['subject_score'] += clf.predict_proba(scmat)[:, 1]
    subinfo_tst1['subject_score'] += clf.predict_proba(scmat_tst1)[:, 1]
    subinfo_tst2['subject_score'] += clf.predict_proba(scmat_tst2)[:, 1]
    subinfo_tst3['subject_score'] += clf.predict_proba(scmat_tst3)[:, 1]
subinfo['subject_score'] /= num_repeat
subinfo_tst1['subject_score'] /= num_repeat
subinfo_tst2['subject_score'] /= num_repeat
subinfo_tst3['subject_score'] /= num_repeat
shap_values1 /= num_repeat
shap_values2 /= num_repeat
shap_values3 /= num_repeat
shap_values1 = pd.DataFrame(shap_values1, index=scmat.index, columns=scmat.columns)
shap_values2 = pd.DataFrame(shap_values2, index=scmat_tst2.index, columns=scmat_tst2.columns)
shap_values3 = pd.DataFrame(shap_values3, index=scmat_tst3.index, columns=scmat_tst3.columns)

#%%
subinfo.to_csv("./c02_subinfo.csv")
subinfo_tst2.to_csv("./seaad_subinfo.csv")
subinfo_tst3.to_csv("./rosmap_subinfo.csv")

shap_values1.to_csv("./c02_shap_values.csv")
shap_values2.to_csv("./seaad_shap_values.csv")
shap_values3.to_csv("./rosmap_shap_values.csv")

feature_importance = np.abs(shap_values1).mean().sort_values(ascending=False)
# feature_importance /= feature_importance.max()
feature_importance.to_csv("./c02_feature_importance.csv")
feature_importance = np.abs(shap_values2).mean().sort_values(ascending=False)
# feature_importance /= feature_importance.max()
feature_importance.to_csv("./seaad_feature_importance.csv")
feature_importance = np.abs(shap_values3).mean().sort_values(ascending=False)
# feature_importance /= feature_importance.max()
feature_importance.to_csv("./rosmap_feature_importance.csv")

#%%
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list('custom', [(0, 'blue'), (1, 'red')])
shap.summary_plot(
    shap_values1.values, 
    scmat.values,  
    feature_names=scmat.columns,
    max_display=10,
    show=False,
    cmap=cmap
    )
plt.savefig(
    './c02_celltype_prioritization.pdf', 
    format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# shap.summary_plot(
#     shap_values2.values, 
#     scmat_tst2.values,  
#     feature_names=scmat_tst2.columns,
#     max_display=10,
#     show=False,
#     cmap=cmap
#     )
# plt.savefig(
#     './seaad_celltype_prioritization_by_SHAP.pdf', 
#     format='pdf', bbox_inches='tight', dpi=600)
# plt.show()

# shap.summary_plot(
#     shap_values3.values, 
#     scmat_tst3.values,  
#     feature_names=scmat_tst3.columns,
#     max_display=10,
#     show=False,
#     cmap=cmap
#     )
# plt.savefig(
#     './rosmap_celltype_prioritization_by_SHAP.pdf', 
#     format='pdf', bbox_inches='tight', dpi=600)
# plt.show()

#%%
from sklearn.metrics import  roc_curve, auc

# # Only for skipping the 100 repeat step and reading directly from stored data
# info_temp1 = adata1.obs[['SubID', 'subject_score']][~adata1.obs['c02_100v100_mask']].drop_duplicates().sort_values(by='SubID', ascending=True)
# info_temp1['SubID'] = info_temp1['SubID'].cat.remove_unused_categories()
# assert (info_temp1['SubID'].values == subinfo_tst1.index).all()
# info_temp2 = adata2.obs[['SubID', 'subject_score']].drop_duplicates().sort_values(by='SubID', ascending=True)
# info_temp2['SubID'] = info_temp2['SubID'].cat.remove_unused_categories()
# assert (info_temp2['SubID'].values == subinfo_tst2.index).all()
# info_temp3 = adata3.obs[['individualID', 'subject_score']].drop_duplicates().sort_values(by='individualID', ascending=True)
# info_temp3['individualID'] = info_temp3['individualID'].cat.remove_unused_categories()
# assert (info_temp3['individualID'].values == subinfo_tst3.index).all()
# subinfo_tst1['subject_score'] = info_temp1['subject_score'].values
# subinfo_tst2['subject_score'] = info_temp2['subject_score'].values
# subinfo_tst3['subject_score'] = info_temp3['subject_score'].values

def plot_roc_auc(y_true, y_pred, color, linestyle, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2, 
             label=f'{label} AUC = {roc_auc:.2f}' if label else f'AUC = {roc_auc:.2f}')

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 25
matplotlib.rcParams['ytick.labelsize'] = 25

fig, ax = plt.subplots(figsize=(15, 15))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plot_roc_auc(
    subinfo_tst1['c02x'].map({'AD': 1, 'Control': 0}),
    subinfo_tst1['subject_score'],
    color='black',
    linestyle='solid',
    label='c02x'
)
plot_roc_auc(
    subinfo_tst2['Cognitive Status'].map({'Dementia': 1, 'No dementia': 0}),
    subinfo_tst2['subject_score'],
    color='black',
    linestyle='dotted',
    label='SEA-AD'
)
plot_roc_auc(
    subinfo_tst3['diagnosis'].map({'AD': 1, 'CTL': 0}),
    subinfo_tst3['subject_score'],
    color='black',
    linestyle=(5, (10, 3)),
    label='ROSMAP'
)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
legend = plt.legend(loc='lower right', title=None, fontsize=30)

plt.savefig('./c02_SEAAD_ROSMAP_donor_score_pred_rocauc.pdf', format='pdf', bbox_inches='tight', dpi=100)
plt.show()






# #%% ############################### braak trend
# matplotlib.rcParams['xtick.labelsize'] = 13
# matplotlib.rcParams['ytick.labelsize'] = 13

# subinfo_tst2['braak_cat'] = subinfo_tst2['Braak'].map({
#     'Braak 0': 'Early',
#     'Braak II': 'Early',
#     'Braak III': 'Inter',
#     'Braak IV': 'Inter',
#     'Braak V': 'Late',
#     'Braak VI': 'Late'
# })

# sns.boxplot(
#     data = subinfo_tst2,
#     # x = 'Braak',
#     x = 'braak_cat',
#     y = 'subject_score',
#     # order=['Braak 0', 'Braak II', 'Braak III', 'Braak IV', 'Braak V', 'Braak VI']
#     order=['Early', 'Inter', 'Late']
# )
# sns.swarmplot(
#     data = subinfo_tst2,
#     # x = 'Braak',
#     x = 'braak_cat',
#     y = 'subject_score',
#     # order=['Braak 0', 'Braak II', 'Braak III', 'Braak IV', 'Braak V', 'Braak VI']
#     order=['Early', 'Inter', 'Late']
# )
# plt.show()

# sns.boxplot(
#     data = subinfo_tst3,
#     x = 'braaksc',
#     y = 'subject_score',
#     order=np.arange(1, 7)
# )
# sns.swarmplot(
#     data = subinfo_tst3,
#     x = 'braaksc',
#     y = 'subject_score',
#     order=np.arange(1, 7)
# )
# plt.show()

# #%%
# adata1.obs['subject_score'] = subinfo.loc[
#     adata1.obs['SubID'].values, 'subject_score'].values
# adata2.obs['subject_score'] = subinfo_tst2.loc[
#     adata2.obs['SubID'].values, 'subject_score'].values
# adata3.obs['subject_score'] = subinfo_tst3.loc[
#     adata3.obs['individualID'].values, 'subject_score'].values

# adata1.write_h5ad(DATA_PATH + 'PsychAD/c02.h5ad')
# adata2.write_h5ad(DATA_PATH + 'SEA-AD/seaad.h5ad')
# adata3.write_h5ad(DATA_PATH + 'ROSMAP/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')









# #%% ################################# Evaluate on RF subject-level
# from sklearn.ensemble import RandomForestClassifier
# import shap

# subid_col = 'SubID'
# cond_col = 'c02x'
# pos_cond = 'AD'
# neg_cond = 'Control'
# class_col = 'subclass'



# np.random.choice(adata2.shape[0], adata2.shape[0]/3, replace=False)

# #
# scmat = PASCode.utils.scmatrix(
#     adata1.obs, 
#     subid_col,
#     class_col=class_col,
#     score_col='pac_score')
# subinfo = PASCode.utils.subject_info(
#     adata1.obs,
#     subid_col,
#     columns=[cond_col])
# scmat = scmat.loc[subinfo.index]

# scmat_trn = PASCode.utils.scmatrix(
#     adata1.obs[adata1.obs['c02_100v100_mask']], 
#     subid_col,
#     class_col=class_col,
#     score_col='pac_score')
# subinfo_trn = PASCode.utils.subject_info(
#     adata1.obs[adata1.obs['c02_100v100_mask']],
#     subid_col,
#     columns=[cond_col])
# scmat_trn = scmat_trn.loc[subinfo_trn.index]

# scmat_tst1 = PASCode.utils.scmatrix(
#     adata1.obs[~adata1.obs['c02_100v100_mask']],
#     subid_col,
#     class_col=class_col,
#     score_col='pac_score')
# subinfo_tst1 = PASCode.utils.subject_info(
#     adata1.obs[~adata1.obs['c02_100v100_mask']],
#     subid_col,
#     columns=[cond_col])
# scmat_tst1 = scmat_tst1.loc[subinfo_tst1.index]

# #
# subid_col = 'SubID'
# cond_col = 'c03x'
# pos_cond = 'AD'
# neg_cond = 'Control'
# scmat_tst2 = PASCode.utils.scmatrix(
#     adata2.obs, 
#     subid_col,
#     class_col=class_col,
#     score_col='pac_score')
# subinfo_tst2 = PASCode.utils.subject_info(
#     adata2.obs,
#     subid_col,
#     columns=[cond_col])
# scmat_tst2 = scmat_tst2.loc[subinfo_tst2.index]

# #
# cond_col = 'diagnosis'
# pos_cond = 'AD'
# neg_cond = 'CTL'
# subid_col = 'individualID'
# scmat_tst3 = PASCode.utils.scmatrix(
#     adata3.obs, 
#     subid_col,
#     class_col=class_col,
#     score_col='pac_score')
# subinfo_tst3 = PASCode.utils.subject_info(
#     adata3.obs,
#     subid_col,
#     columns=[cond_col])
# scmat_tst3 = scmat_tst3.loc[subinfo_tst3.index]


# #%%
# # 1. transductive > inductive? (c02 left > c03/rosmap). TODO val by using 11v11. TODO if true, try using transductive
# # 2. too few donors for c03 (only 9v9), 4 tools are unstable for few donors (TODO if this is true, abandon c03, use datasets that have more donors, e.g., seaAD)
# # 3. inherent differeneces between c02 and c03 (just bad data / or, bad batch correction of harmnoy)
# # 4. feature selection, e.g., top 10 cell types 
# # 5. the essential cause is either
# #   5.1 their data is bad? TODO try seaAD; try c02 self inductive.
# #   5.2 our model is not good enough (but we don't want to go back... we already told kalpana that was the last time for a change)
# # 6. PAC level prediction is not good in the first place. 
# #   TODO experimenting using simulated good PACs to see if it improves?
# # NOTE: worse 7. try MLP predictor instead of RF. .
# # 8. 

# adata1.obs['pac_score'][adata1.obs['c02_11v11_mask']]
# adata1.obs['rra_pac'][adata1.obs['c02_11v11_mask']]

# sns.boxplot(
#     x=adata1.obs['rra_pac'][adata1.obs['c02_11v11_mask']],
#     y=adata1.obs['pac_score'][adata1.obs['c02_11v11_mask']],
# )

# ################################################################################
# ################################################################################
# ################################################################################
# #%% [markdown]
# ### cell type prioritization by mean shap values across 100 runs
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier

# shap_top = ['Micro', 'Astro', 'EN_L3_5_IT_2', 'EN_L2_3_IT', 'VLMC', 'Oligo', 'IN_SST', 'IN_PVALB', 'IN_VIP', 'EN_L3_5_IT_3']

# num_repeat = 10

# cond_col = 'c02x'
# pos_cond = 'AD'
# neg_cond = 'Control'
# shap_values = np.zeros_like(scmat.values)
# subinfo['subject_score'] = 0.0
# subinfo_tst1['subject_score'] = 0.0
# subinfo_tst2['subject_score'] = 0.0
# subinfo_tst3['subject_score'] = 0.0
# for seed in np.arange(num_repeat):
#     clf = RandomForestClassifier(n_estimators=100, random_state=seed)
#     clf.fit(X=scmat_trn, 
#             y=subinfo_trn[cond_col].map({pos_cond: 1, neg_cond: 0}))
#     explainer = shap.TreeExplainer(clf)
#     shap_values += explainer.shap_values(scmat.values)[1]

#     subinfo['subject_score'] += clf.predict_proba(scmat)[:, 1]
#     subinfo_tst1['subject_score'] += clf.predict_proba(scmat_tst1)[:, 1]
#     subinfo_tst2['subject_score'] += clf.predict_proba(scmat_tst2)[:, 1]
#     subinfo_tst3['subject_score'] += clf.predict_proba(scmat_tst3)[:, 1]
# subinfo['subject_score'] /= num_repeat
# subinfo_tst1['subject_score'] /= num_repeat
# subinfo_tst2['subject_score'] /= num_repeat
# subinfo_tst3['subject_score'] /= num_repeat
# shap_values /= num_repeat

# #%%
# shap_values = pd.DataFrame(shap_values, index=scmat.index, columns=scmat.columns)
# shap.summary_plot(
#     shap_values.values, 
#     scmat.values,  
#     feature_names=scmat.columns,
#     max_display=10,
#     show=False
#     )

# plt.savefig(
#     './c02_celltype_prioritization_by_SHAP_temp.pdf', 
#     format='pdf', bbox_inches='tight', dpi=100)
# plt.show()

# shap_values.to_csv("./c02_shap_values_temp.csv")

# #%%
# from sklearn.metrics import  roc_curve, auc

# # # Only for skipping the 100 repeat step and reading directly from stored data
# # info_temp1 = adata1.obs[['SubID', 'subject_score']][~adata1.obs['c02_100v100_mask']].drop_duplicates().sort_values(by='SubID', ascending=True)
# # info_temp1['SubID'] = info_temp1['SubID'].cat.remove_unused_categories()
# # assert (info_temp1['SubID'].values == subinfo_tst1.index).all()
# # info_temp2 = adata2.obs[['SubID', 'subject_score']].drop_duplicates().sort_values(by='SubID', ascending=True)
# # info_temp2['SubID'] = info_temp2['SubID'].cat.remove_unused_categories()
# # assert (info_temp2['SubID'].values == subinfo_tst2.index).all()
# # info_temp3 = adata3.obs[['individualID', 'subject_score']].drop_duplicates().sort_values(by='individualID', ascending=True)
# # info_temp3['individualID'] = info_temp3['individualID'].cat.remove_unused_categories()
# # assert (info_temp3['individualID'].values == subinfo_tst3.index).all()
# # subinfo_tst1['subject_score'] = info_temp1['subject_score'].values
# # subinfo_tst2['subject_score'] = info_temp2['subject_score'].values
# # subinfo_tst3['subject_score'] = info_temp3['subject_score'].values

# def plot_roc_auc(y_true, y_pred, color, linestyle, label=None):
#     fpr, tpr, _ = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2, 
#              label=f'{label} AUC = {roc_auc:.2f}' if label else f'AUC = {roc_auc:.2f}')

# import matplotlib as mpl
# mpl.rcParams['xtick.labelsize'] = 25
# mpl.rcParams['ytick.labelsize'] = 25

# fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
# plot_roc_auc(
#     subinfo_tst1['c02x'].map({'AD': 1, 'Control': 0}),
#     subinfo_tst1['subject_score'],
#     color='black',
#     linestyle='solid',
#     label='c02x'
# )
# plot_roc_auc(
#     subinfo_tst2['c03x'].map({'AD': 1, 'Control': 0}),
#     subinfo_tst2['subject_score'],
#     color='black',
#     linestyle='dotted',
#     label='c03x'
# )
# plot_roc_auc(
#     subinfo_tst3['diagnosis'].map({'AD': 1, 'CTL': 0}),
#     subinfo_tst3['subject_score'],
#     color='black',
#     linestyle=(5, (10, 3)),
#     label='ROSMAP'
# )
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])

# # yticks = ax.yaxis.get_major_ticks()
# # if yticks:
# #     yticks[0].label1.set_visible(False)
# # yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
# # if yticklabels:  # Make sure our list is not empty
# #     yticklabels[0] = ''
# #     ax.set_yticklabels(yticklabels)

# plt.xlabel('False Positive Rate', fontsize=30)
# plt.ylabel('True Positive Rate', fontsize=30)
# # plt.title('Receiver Operating Characteristic (ROC)', fontsize=20)
# legend = plt.legend(loc='lower right', title=None, fontsize=30)

# plt.savefig('./c02_c03_rosmap_subject_roc_auc_temp.pdf', format='pdf', bbox_inches='tight', dpi=100)
# plt.show()





# %%
###############################################################################
# backup code
###############################################################################


tab = adata2.obs.groupby(['Subclass','subclass']).size().unstack()
ntab2 = tab.div(tab.sum(axis=1), axis=0)
tab = adata3.obs.groupby(['Subcluster','subclass']).size().unstack()
ntab3 = tab.div(tab.sum(axis=1), axis=0)

ids = ntab2.idxmax(axis=1)
subclass2_palette = {ids.index[i] : subclass_palette[k] for i, k in enumerate(ids.values)}
subclass2 = ['Astrocyte',
             'Microglia-PVM',
             'Oligodendrocyte',
             'OPC',
             'L2/3 IT',
             'L4 IT',
             'L5 IT',
             'L5/6 NP',
             'L5 ET',
             'L6b',
             'L6 CT',
             'L6 IT',
             'L6 IT Car3',
             'Lamp5 Lhx6',
             'Lamp5',
             'Pvalb',
             'Sst',
             'Sst Chodl',
             'Vip',
             'Chandelier',
             'Endothelial',
             'Pax6',
             'Sncg',
             'VLMC'
             ]
subclass2_palette = {k: subclass2_palette[k] for k in subclass2}



ids = ntab3.idxmax(axis=1)
subclass3_palette = {ids.index[i] : subclass_palette[k] for i, k in enumerate(ids.values)}
subclass3 = ['Ast0', 'Ast1', 'Ast2', 'Ast3', 'Mic0', 'Mic1', 'Mic2', 'Mic3', 
             'Oli0', 'Oli1', 'Oli3', 'Oli4', 'Oli5', 
             'Opc0', 'Opc1', 'Opc2', 
             'Ex0', 'Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6', 'Ex7', 'Ex8', 'Ex9', 'Ex11', 'Ex12', 'Ex14', 
             'In0', 'In1', 'In2', 'In3', 'In4', 'In5', 'In6', 'In7', 'In8', 'In9', 'In10', 'In11', 
             'End1', 'End2', 'Per']
subclass3_palette = {k: subclass3_palette[k] for k in subclass3}

subclass_palette





subclass3_palette = {
    'Ast0': '#C19A6B',
    'Ast1': '#C19A6B',
    'Ast2': '#C19A6B',
    'Ast3': '#C19A6B',
    'Mic0': '#F75D59',
    'Mic1': '#F75D59',
    'Mic2': '#F75D59',
    'Mic3': '#F75D59',
    'Oli0': '#ECE5B6',
    'Oli1': '#ECE5B6',
    'Oli3': '#ECE5B6',
    'Oli4': '#ECE5B6',
    'Oli5': '#ECE5B6',
    'Opc0': '#FFF380',
    'Opc1': '#FFF380',
    'Opc2': '#FFF380',
    'Ex0': '#659EC7',
    'Ex1': '#659EC7',
    'Ex2': '#659EC7',
    'Ex3': '#659EC7',
    'Ex4': '#659EC7',
    'Ex5': '#659EC7',
    'Ex6': '#659EC7',
    'Ex7': '#659EC7',
    'Ex8': '#659EC7',
    'Ex9': '#659EC7',
    'Ex11': '#659EC7',
    'Ex12': '#659EC7',
    'Ex14': '#659EC7',
    'In0': '#4E8975',
    'In1': '#4E8975',
    'In2': '#4E8975',
    'In3': '#4E8975',
    'In4': '#4E8975',
    'In5': '#4E8975',
    'In6': '#4E8975',
    'In7': '#4E8975',
    'In8': '#4E8975',
    'In9': '#4E8975',
    'In10': '#4E8975',
    'In11': '#4E8975',
    'End1': '#FFA62F',
    'End2': '#FFA62F',
    'Per': '#A52A2A'
}
