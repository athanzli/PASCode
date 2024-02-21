# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('../..')
import PASCode

import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PASCode.utils import *

#%%
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

adata = sc.read_h5ad(DATA_PATH + 'PsychAD/r01_only_obs_obsm.h5ad')

#%%
###############################################################################
# remove 19k and redo cell type label transferring
###############################################################################
tormv = pd.read_csv("/home/che82/athan/PASCode/240124_PsychAD_freeze3_outlier_nuclei.csv", index_col=0)
adata = adata[~adata.obs.index.isin(tormv.index)]

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
# umap
###############################################################################
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='model_layer2_umap',
    class_col='subclass', 
    class_palette=subclass_palette,
    text_on_plot=True,
    save_path='./r01_latent_subclass_umap.tiff'
)
plot_legend(
    handles, labels, 2, "../PsychAD_subclass_legend_nonum.tiff")
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='model_layer2_umap',
    class_col='subclass', 
    class_palette=subclass_palette,
    text_on_plot=False,
    save_path='./r01_latent_subclass_umap_nolabel.tiff'
)
plot_pac_umap(
    adata, 
    umap_key='model_layer2_umap', 
    pac_col='pac_score',
    save_path='./r01_latent_pac_umap.tiff',
    colors=['#1f7a0f', '#ffffff', '#591496']
)


# %%
###############################################################################
# 
###############################################################################
subinfo_all = pd.read_csv('/home/che82/data/psychAD/metadata.csv', index_col=0)

# NOTE specify column names
cond_col = 'r01x_bi'
pos_cond = 'bk6'
neg_cond = 'bk0'
subid_col = 'SubID'
score_col = 'pac_score'
class_col = 'subclass'
trn_sub = adata.obs[adata.obs['r01_30v30_mask']]['SubID'].unique()

adata.obs['c28x'] = subinfo_all.loc[adata.obs[subid_col].values, 'c28x'].values
adata.obs['c02x'] = subinfo_all.loc[adata.obs[subid_col].values, 'c02x'].values
# adata.obs['c92x'] = subinfo_all.loc[adata.obs[subid_col].values, 'c92x'].values
# adata.obs['c125x'] = subinfo_all.loc[adata.obs[subid_col].values, 'c125x'].values
adata.obs['Ethnicity'] = subinfo_all.loc[adata.obs[subid_col].values, 'Ethnicity'].values

scmat = PASCode.utils.scmatrix(
    adata.obs, 
    subid_col=subid_col, 
    class_col=class_col, 
    score_col=score_col)
subinfo = PASCode.utils.subject_info(
    adata.obs, 
    subid_col, 
    columns=[cond_col, 'Sex', 'r01x', 'c28x', 'Ethnicity','c02x']) # 'c92x','c125x'])
scmat = scmat.loc[subinfo.index]
assert (scmat.index==subinfo.index).all()
trn_mask = scmat.index.isin(trn_sub)

# #%%###############HCF playground start
# from sklearn.manifold import TSNE

# #import umap
# #reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=5)
# from sklearn.decomposition import KernelPCA
# #from sklearn.decomposition import PCA

# #pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.7)
# #emb = reducer.fit_transform(scmat)
# emb = TSNE(n_components=2, learning_rate='auto').fit_transform(scmat)
# #emb = pca.fit_transform(scmat)
# assert (scmat.index==subinfo.index).all()

# #sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=subinfo['r01x'],palette="viridis")
# #sns.scatterplot(x=emb[subinfo['c28x'] != 'AD_resilient',0], y=emb[subinfo['c28x'] != 'AD_resilient',1], hue=subinfo[subinfo['c28x'] != 'AD_resilient']['r01x'],palette="viridis")
# sns.boxplot(x=subinfo['r01x'], y=emb[:,0],palette="viridis")
# sns.swarmplot(x=subinfo['r01x'], y=emb[:,0],color='black',size=3)
# #############HCF playground end

# %%
import shap
import sklearn
import numpy as np
num_repeat = 100

subinfo['subject_score'] = 0
shap_values = np.zeros(scmat.shape)
for seed in np.arange(num_repeat):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=seed)
    X = scmat[trn_mask].values
    y = subinfo[trn_mask][cond_col].map({pos_cond:1, neg_cond:0}).values
    clf.fit(X, y)
    subinfo['subject_score'] += clf.predict_proba(scmat)[:,-1]
    shap_values += shap.TreeExplainer(clf).shap_values(scmat.values)[1]
subinfo['subject_score'] /= num_repeat
shap_values /= num_repeat

shap.summary_plot(shap_values, scmat, max_display=10, show=False)
plt.savefig('./shap_summary_plot.pdf', dpi=100)

adata.obs['subject_score'] = subinfo.loc[adata.obs[subid_col].values, 'subject_score'].values

shap_values = pd.DataFrame(shap_values, columns=scmat.columns, index=scmat.index)
shap_values.to_csv("./shap_values.csv") # shap_values = pd.read_csv("./shap_values.csv", index_col=0)
subinfo.to_csv("./subinfo.csv") # subinfo = pd.read_csv("./subinfo.csv", index_col=0)

#%% braak stages
subinfo = pd.read_csv("./subinfo.csv", index_col=0)
shap_values = pd.read_csv("./shap_values.csv", index_col=0)

import numpy as np
import rpy2

# colors = ['#0015b0', '#4405e3', '#7f05e3', '#8f008f', '#ad0258', '#E60000', '#B20000']
# colors = ['#004C99', '#223E7F', '#463265', '#6A254B', '#8E1831', '#B20B17', '#CC0000']\
colors = [
    "#1f7a0f",
    "#389e26",
    "#6dc95d",
    "#7389d1",
    "#4969d1",
    "#9e61d4",
    "#591496" 
]

def plot_braak_stages(subinfo, colors, mask, save_path):
    sns.boxplot(data=subinfo[mask], x='r01x',y='subject_score', palette=colors)
    sns.swarmplot(data=subinfo[mask], x='r01x',y='subject_score',size=3,color='black')
    plt.xlabel('Braak stages')
    plt.xticks([0,1,2,3,4,5,6], ['0','1','2','3','4','5','6'])
    plt.ylabel('Donor AD progression stage time') # NOTE
    plt.savefig(save_path, dpi=100)
    plt.show()

    # for test
    groups = subinfo['r01x'].values[mask] # NOTE this is the braak stage
    order_idx = np.argsort(groups)
    groups = groups[order_idx].astype(int)
    group_data = subinfo['subject_score'].values[mask] # NOTE this is the donor score
    group_data = group_data[order_idx]

    # Jonckheere-Terpstra test
    clinfun = rpy2.robjects.packages.importr('clinfun')
    rpy2.robjects.numpy2ri.activate()
    ## NOTE uncomment these three lines and use it for "jonckheere_test" if the P value is stuck at 0.0002
    # nperm = None
    # if len(group_data) > 300:
    #     nperm = 5000
    jt_res = clinfun.jonckheere_test(
        x=rpy2.robjects.FloatVector(group_data),
        g=rpy2.robjects.IntVector(groups), 
        alternative='increasing',
    )
    print(f'Jonckheere-Terpstra test P-value: {jt_res[1].item()}')

    # Fit a regression model
    import statsmodels.api as sm
    X = sm.add_constant(groups)
    model = sm.OLS(group_data, X)
    results = model.fit()
    print('Linear regression R^2 value:', results.rsquared)

# all donors within r01x
plot_braak_stages(
    subinfo, 
    colors, 
    mask=np.full(subinfo.shape[0], True),
    save_path='./r01_braakstages_all.pdf'
)

# all donors within r01x except for AD res
mask = subinfo['c28x'] != 'AD_resilient'
plot_braak_stages(
    subinfo, 
    colors, 
    mask,
    save_path='./r01_braakstages_wo_ad_res.pdf'
)

# braak stages only retaining those with c28 labels
mask = ~subinfo['c28x'].isna()
plot_braak_stages(
    subinfo, 
    colors, 
    mask,
    save_path='./r01_braakstages_only_c28_donors.pdf'
)

# braak stages removing AD_resilient within c28
mask = (subinfo['c28x']!='AD_resilient') & (~subinfo['c28x'].isna())
plot_braak_stages(
    subinfo, 
    colors, 
    mask,
    save_path='./r01_braakstages_c28_wo_resilient_donors.pdf'
)

# %%
###############################################################################
# braak hue by c28x
###############################################################################
mask = ~subinfo['c28x'].isna()
hue_order = ['Control', 'AD_resilient','AD_strict']
my_palette = {'Control':"#1f7a0f", 'AD_resilient':"#20b8da", 'AD_strict':"#591496"}
sns.boxplot(x='r01x', y='subject_score', hue='c28x', data=subinfo[mask],
            palette=my_palette, hue_order=hue_order)
sns.swarmplot(x='r01x', y='subject_score', hue='c28x', data=subinfo[mask],
              dodge=True, size=3, hue_order=hue_order, legend=False, color='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('r01x braak stages')
plt.xticks([0,1,2,3,4,5,6], ['0','1','2','3','4','5','6'])
plt.ylabel('Subject phenotype score')
plt.savefig('./c28x_across_r01x_braak.pdf', dpi=100)

##  test
from scipy.stats import mannwhitneyu, ttest_ind 

for brk in [3,4,5,6]:
    grp1 = subinfo[(subinfo['r01x'] == brk) & (subinfo['c28x'] == 'AD_strict')]['subject_score'].values
    grp2 = subinfo[(subinfo['r01x'] == brk) & (subinfo['c28x'] == 'AD_resilient')]['subject_score'].values
    stats = mannwhitneyu(grp1, grp2)
    # stats = ttest_ind(grp1, grp2, equal_var=False) # Welch's t-test
    # stats = ttest_ind(grp1, grp2, equal_var=True)
    print(stats.pvalue)

mask = (subinfo['r01x'] == 4)# | (subinfo['r01x'] == 5)
grp1 = subinfo[mask & (subinfo['c28x'] == 'AD_strict')]['subject_score'].values
grp2 = subinfo[mask & (subinfo['c28x'] == 'AD_resilient')]['subject_score'].values
stats = mannwhitneyu(grp1, grp2)
# stats = ttest_ind(grp1, grp2, equal_var=False) # Welch's t-test
# stats = ttest_ind(grp1, grp2, equal_var=True)
print(stats.pvalue)


# %%
###############################################################################
# c28 cats with subject scores
###############################################################################
mask = ~subinfo['c28x'].isna()
order = ['Control', 'AD_resilient','AD_strict']
my_palette = {'Control':"#1f7a0f", 'AD_resilient':"#20b8da", 'AD_strict':"#591496"}
sns.boxplot(x='c28x', y='subject_score', data=subinfo[mask],
            palette=my_palette, order=order)
sns.swarmplot(x='c28x', y='subject_score', data=subinfo[mask], order=order,
              dodge=True, size=3, legend=False, color='black')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('')
plt.ylabel('Subject phenotype score')
plt.savefig('./c28x_boxplot.pdf', dpi=100)

# wilcoxon test
from scipy.stats import mannwhitneyu
grp1 = subinfo[(subinfo['c28x'] == 'AD_strict')]['subject_score'].values
grp2 = subinfo[(subinfo['c28x'] == 'AD_resilient')]['subject_score'].values
stats = mannwhitneyu(grp1, grp2)
print(stats.pvalue)


# #%%
# ###############################################################################
# # same as the previous one but with only >= braak stages
# ###############################################################################
# mask = (~subinfo['c28x'].isna()) & (subinfo['r01x'] > 3)
# order = ['Control', 'AD_resilient','AD_strict']
# my_palette = {'Control':"#1f7a0f", 'AD_resilient':"#20b8da", 'AD_strict':"#591496"}
# sns.boxplot(x='c28x', y='subject_score', data=subinfo[mask],
#             palette=my_palette, order=order)
# sns.swarmplot(x='c28x', y='subject_score', data=subinfo[mask], order=order,
#               dodge=True, size=3, legend=False, color='black')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xlabel('')
# plt.ylabel('Subject phenotype score')
# plt.savefig('./c28x_boxplot.pdf', dpi=100)

# # wilcoxon test
# from scipy.stats import mannwhitneyu
# grp1 = subinfo[(subinfo['c28x'] == 'AD_strict')]['subject_score'].values
# grp2 = subinfo[(subinfo['c28x'] == 'AD_resilient')]['subject_score'].values
# stats = mannwhitneyu(grp1, grp2)
# print(stats.pvalue)

# #%% ############################ EN_L3_5_IT_3
# subinfo['subject_score_EN_L3_5_IT_3'] = shap_values['EN_L3_5_IT_3']

# mask = ~subinfo['c28x'].isna()
# order = ['Control', 'AD_resilient','AD_strict']
# my_palette = {'Control':"#1f7a0f", 'AD_resilient':"#DAA520", 'AD_strict':"#591496"}
# sns.boxplot(x='c28x', y='subject_score_EN_L3_5_IT_3', data=subinfo[mask],
#             palette=my_palette, order=order)
# sns.swarmplot(x='c28x', y='subject_score_EN_L3_5_IT_3', data=subinfo[mask], order=order,
#               dodge=True, size=3, legend=False, color='black')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xlabel('')
# plt.ylabel('Subject phenotype score')
# plt.savefig('./c28x_boxplot_EN_L3_5_IT_3.pdf', dpi=100)

#%% 
###############################################################################
# which celltypes differ AD_resilient and AD_strict bubble plot
###############################################################################
scmat = scmat.loc[subinfo.index]
scmat = scmat.loc[:, list(subclass_palette.keys())]
ctps = scmat.columns.astype(str).values
shap_mat = pd.DataFrame(shap_values, columns=scmat.columns, index=scmat.index)

## calculate p value
from scipy.stats import mannwhitneyu
pvs = []  #p-values
lfcs = []  #differences
for ctp in ctps:
    #stas = mannwhitneyu(shap_mat[(subinfo['c28x']=='AD_resilient') & (subinfo['r01x']>3)][ctp], 
    #                   shap_mat[(subinfo['c28x']=='AD_strict') & (subinfo['r01x']>3)][ctp])
    stas = mannwhitneyu(shap_mat[(subinfo['c28x']=='AD_resilient')][ctp], 
                        shap_mat[(subinfo['c28x']=='AD_strict')][ctp])
    pvs.append(stas.pvalue)
    #lfc = np.log2(np.mean(scmat[(subinfo['c28x']=='AD_strict')][ctp])/np.mean(scmat[(subinfo['c28x']=='AD_resilient')][ctp]))
    #lfc = np.log2(np.median(scmat[(subinfo['c28x']=='AD_strict')][ctp])/np.median(scmat[(subinfo['c28x']=='AD_resilient')][ctp]))
    lfc = np.median(scmat[(subinfo['c28x']=='AD_strict')][ctp]) - \
        np.median(scmat[(subinfo['c28x']=='AD_resilient')][ctp])
    lfcs.append(lfc)

## with bubble plot showing p values and log fold changes
# sns.scatterplot(
#     x = np.array(lfcs)[np.argsort(pvs)],
#     y = -np.log10(np.array(pvs)[np.argsort(pvs)]), 
# )
# for i in range(len(lfcs)):
#     plt.text(
#         np.array(lfcs)[np.argsort(pvs)][i], 
#         -np.log10(np.array(pvs)[np.argsort(pvs)])[i], 
#         ctps[np.argsort(pvs)][i], 
#         horizontalalignment='left', 
#         size='medium', 
#         color='black', 
#         weight='semibold'
#     )
from statsmodels.stats.multitest import multipletests
padj = multipletests(pvs, method='fdr_bh')[1]

data = {
    'X': np.array(lfcs)[np.argsort(pvs)],
    'minus_log10_padj': -np.log10(padj[np.argsort(pvs)]),
    'padj': padj[np.argsort(pvs)],
    'Y': -np.log10(np.array(pvs)[np.argsort(pvs)]),
    'AD_strict_median': np.median(scmat[(subinfo['c28x']=='AD_strict')][ctps[np.argsort(pvs)]], axis=0),
    'AD_resilient_median': np.median(scmat[(subinfo['c28x']=='AD_resilient')][ctps[np.argsort(pvs)]], axis=0),
    'label': ctps[np.argsort(pvs)],
    'size':  -np.log10(np.array(pvs)[np.argsort(pvs)])*200,
    #'size':  -np.log10(np.array(minus_log10_padj)[np.argsort(minus_log10_padj)])*200,
    'color': pd.DataFrame(
        list(subclass_palette.items()), 
        columns=['Category', 'Color']).iloc[np.argsort(pvs),:]['Color'].values
}

df = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df['X'], 
    #df['minus_log10_padj'],
    df['Y'], 
    s=df['size'], 
    color=df['color'], 
    alpha=0.5, 
    edgecolors="white", 
    linewidth=0.5,
)
for i, txt in enumerate(df['label']):
    ax.annotate(txt, (df['X'][i], df['Y'][i]), fontsize=12, ha='center')
    #ax.annotate(txt, (df['X'][i], df['minus_log10_padj'][i]), fontsize=12, ha='center')
ax.set_title('AD-resilient VS AD-strict', fontsize=13)
#ax.set_xlabel('log2 fold change',fontsize=13)
ax.set_xlabel('AD_strict_median - AD_resilient_median',fontsize=13)
ax.set_ylabel('-log10 P-value',fontsize=13)
plt.savefig('AD-resVSstrict_bubble.pdf', format="pdf", bbox_inches="tight", dpi=600)
plt.show()

df.index = df['label']
df.drop(columns=['label', 'size', 'color'], inplace=True)
df.rename(columns={'X':'AD_strict_median - AD_resilient_median'}, inplace=True)
df.to_csv('./AD_res_vs_strict_bubble.csv')

#%%
## with barplot showing p value
# sns.barplot(x = - np.log10(np.array(pvs)[np.argsort(pvs)]), 
#             y = ctps[np.argsort(pvs)]); plt.show()

######################################################################




# #%% ########################################## r02 NOTE bad results... ##########################################
# adata_tst = sc.read_h5ad('/home/che82/data/psychAD/contrasts/r02.h5ad')
# adata_tst.obs.columns

# import torch
# model = PASCode.model.GAT(heads=4, out_channels=64, in_channels=adata_tst.shape[1], num_class=3)
# model.load_state_dict(torch.load('/home/che82/athan/PASCode/code/github_repo/training/trained_models/r01_model_with_cna.pt'))
# adata_tst.obs['pac_score'] = model.predict(PASCode.model.Data().adata2gdata(adata_tst))

# subid_col = 'SubID'
# score_col = 'pac_score'
# class_col = 'subclass'

# adata_tst.obs['c29x'] = subinfo_all.loc[adata_tst.obs[subid_col].values, 'c29x'].values
# adata_tst.obs['Ethnicity'] = subinfo_all.loc[adata_tst.obs[subid_col].values, 'Ethnicity'].values

# scmat = PASCode.utils.scmatrix(adata_tst.obs, 
#                                subid_col=subid_col, 
#                                class_col=class_col, 
#                                score_col=score_col)
# subinfo = PASCode.utils.subject_info(adata_tst.obs, subid_col, 
#                                          columns=[ 'Sex', 'r02x', 'c29x', 'Ethnicity'])
# assert (scmat.index==subinfo.index).all()
# subinfo['subject_score'] = clf.predict_proba(scmat)[:,-1]


# import matplotlib.pyplot as plt
# hue_order = ['Control','AD_resilient','AD_strict']
# sns.boxplot(x=subinfo['r02x'], y=subinfo['subject_score'], hue=subinfo['c29x'],palette=["#FFA7A0", "#ABEAC9","red"],hue_order=hue_order)
# sns.swarmplot(x=subinfo['r02x'], y=subinfo['subject_score'], hue=subinfo['c29x'],dodge=True,size=5,hue_order=hue_order, legend=False)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# mask = ~subinfo['c29x'].isna()
# sns.boxplot(x='c29x', y='subject_score', data=subinfo[mask], order=hue_order)
# sns.swarmplot(x='c29x', y='subject_score', size=3, color='black', data=subinfo[mask], order=hue_order)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# # %% 
# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(scmat.values)
# shap.summary_plot(shap_values[1], scmat, max_display=10)

# #%% braak stages
# import numpy as np
# colors = ['#0015b0', '#4405e3', '#7f05e3', '#8f008f', '#ad0258', '#E60000', '#B20000']
# # colors = ['#004C99', '#223E7F', '#463265', '#6A254B', '#8E1831', '#B20B17', '#CC0000']

# sns.boxplot(data=subinfo, x='r02x',y='subject_score', palette=colors)
# sns.swarmplot(data=subinfo, x='r02x',y='subject_score',size=3,color='black'); plt.show()

# # for test
# groups = subinfo['r02x'].values
# order_idx = np.argsort(groups)
# groups = groups[order_idx].astype(int)
# group_data = subinfo['subject_score'].values
# group_data = group_data[order_idx]

# # Jonckheere-Terpstra test
# import rpy2
# clinfun = rpy2.robjects.packages.importr('clinfun')
# rpy2.robjects.numpy2ri.activate()
# nperm = None
# # if len(group_data) > 300:
# #     nperm = 5000
# jt_res = clinfun.jonckheere_test(x=rpy2.robjects.FloatVector(group_data),
#                                 g=rpy2.robjects.IntVector(groups), 
#                                 alternative='increasing', 
#                                 #nperm=nperm
#                                 )
# print(f'JT test P-value: {jt_res[1].item():.3e}')

# # Fit the regression model
# import statsmodels.api as sm
# X = sm.add_constant(groups)
# model = sm.OLS(group_data, X)
# results = model.fit()
# print('LR R^2 value:', results.rsquared)

# #%% braak stages only retaining those with c28 labels
# mask = ~subinfo['c29x'].isna()
# sns.boxplot(data=subinfo[mask], x='r02x',y='subject_score', palette=colors)
# sns.swarmplot(data=subinfo[mask], x='r02x',y='subject_score',size=3,color='black'); plt.show()

# # for test
# groups = subinfo['r02x'].values[mask]
# order_idx = np.argsort(groups)
# groups = groups[order_idx].astype(int)
# group_data = subinfo['subject_score'].values[mask]
# group_data = group_data[order_idx]

# # Jonckheere-Terpstra test
# import rpy2
# clinfun = rpy2.robjects.packages.importr('clinfun')
# rpy2.robjects.numpy2ri.activate()
# # nperm = None
# # if len(group_data) > 300:
# #     nperm = 5000
# jt_res = clinfun.jonckheere_test(x=rpy2.robjects.FloatVector(group_data),
#                                 g=rpy2.robjects.IntVector(groups), 
#                                 alternative='increasing', 
#                                 # nperm=nperm
#                                 )
# print(f'JT test P-value: {jt_res[1].item():.3e}')

# # Fit the regression model
# import statsmodels.api as sm
# X = sm.add_constant(groups)
# model = sm.OLS(group_data, X)
# results = model.fit()
# print('LR R^2 value:', results.rsquared)

# #%% braak stages removing AD_resilient within c28
# mask = (subinfo['c29x']!='AD_resilient') & (~subinfo['c29x'].isna())
# sns.boxplot(data=subinfo[mask], x='r02x',y='subject_score', palette=colors)
# sns.swarmplot(data=subinfo[mask], x='r02x',y='subject_score',size=3,color='black'); plt.show()

# # for test
# groups = subinfo['r02x'].values[mask]
# order_idx = np.argsort(groups)
# groups = groups[order_idx].astype(int)
# group_data = subinfo['subject_score'].values[mask]
# group_data = group_data[order_idx]

# # Jonckheere-Terpstra test
# import rpy2
# clinfun = rpy2.robjects.packages.importr('clinfun')
# rpy2.robjects.numpy2ri.activate()
# nperm = None
# # if len(group_data) > 300:
# #     nperm = 5000
# jt_res = clinfun.jonckheere_test(x=rpy2.robjects.FloatVector(group_data),
#                                 g=rpy2.robjects.IntVector(groups), 
#                                 alternative='increasing', 
#                                 # nperm=nperm
#                                 )
# print(f'JT test P-value: {jt_res[1].item():.3e}')

# # Fit the regression model
# import statsmodels.api as sm
# X = sm.add_constant(groups)
# model = sm.OLS(group_data, X)
# results = model.fit()
# print('LR R^2 value:', results.rsquared)

# # %%
# import matplotlib.pyplot as plt
# hue_order = ['Control','AD_resilient','AD_strict']
# sns.boxplot(x=subinfo['r02x'], y=subinfo['subject_score'], hue=subinfo['c29x'],palette=["#FFA7A0", "#ABEAC9","red"],hue_order=hue_order)
# sns.swarmplot(x=subinfo['r02x'], y=subinfo['subject_score'], hue=subinfo['c29x'],dodge=True,size=5,hue_order=hue_order, legend=False)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# mask = ~subinfo['c29x'].isna()
# sns.boxplot(x='c29x', y='subject_score', data=subinfo[mask], order=hue_order)
# sns.swarmplot(x='c29x', y='subject_score', size=3, color='black', data=subinfo[mask], order=hue_order)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# #%% which celltypes differ AD_resilient and AD_strict
# ctps = scmat.columns.astype(str).values
# pvals_res = dict(keys=ctps, values=np.zeros(len(ctps)))

# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(scmat.values)

# shap_mat = pd.DataFrame(shap_values[1], columns=scmat.columns, index=scmat.index)

# from scipy.stats import mannwhitneyu
# pvs = []
# for ctp in ctps:
#     stas = mannwhitneyu(shap_mat[(subinfo['c29x']=='AD_resilient')][ctp], 
#                         shap_mat[(subinfo['c29x']=='AD_strict')][ctp])
#     pvs.append(stas.pvalue)
# # sns.barplot(x=-np.log10(pvs)[np.argsort(pvs)], y = ctps[np.argsort(pvs)]); plt.show()

# sns.barplot(x = - np.log10(np.array(pvs)[np.argsort(pvs)]), 
#             y = ctps[np.argsort(pvs)]); plt.show()

# # %%































# #%% which celltypes differ AD_resilient and AD_strict
# ctps = scmat.columns.astype(str).values
# pvals_res = dict(keys=ctps, values=np.zeros(len(ctps)))

# clf = sklearn.ensemble.RandomForestClassifier(n_estimators=5000, random_state= 3475  )
# X = scmat[trn_mask].values
# y = subinfo[trn_mask][cond_col].map({pos_cond:1, neg_cond:0}).values
# clf.fit(X, y)
# subinfo['subject_score'] = clf.predict_proba(scmat)[:,-1]

# mask = np.full(subinfo.shape[0], True)
# subinfo0 = subinfo[mask] # remove nan
# scmat0 = scmat[mask]

# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(scmat0.values)

# shap_mat = pd.DataFrame(shap_values[1], columns=scmat0.columns, index=scmat0.index)

# from scipy.stats import mannwhitneyu
# pvs = []
# for ctp in ctps:
#     stas = mannwhitneyu(shap_mat[(subinfo0['c29x']=='AD_resilient')][ctp], 
#                         shap_mat[(subinfo0['c29x']=='AD_strict')][ctp])
#     pvs.append(stas.pvalue)
# # sns.barplot(x=-np.log10(pvs)[np.argsort(pvs)], y = ctps[np.argsort(pvs)]); plt.show()

# sns.barplot(x = - np.log10(np.array(pvs)[np.argsort(pvs)]), 
#             y = ctps[np.argsort(pvs)]); plt.show()





# # %% 
# mask = ~subinfo['r01x_bi'].isna()

# # Prepare a figure with as many subplots as there are columns in 'scmat'
# num_columns = len(scmat.columns)
# fig, axes = plt.subplots(num_columns, 1, figsize=(10, 6 * num_columns))  # Adjust the size as needed

# for i, (col, ax) in enumerate(zip(scmat.columns, axes.flatten())):
#     # For each column, create a swarmplot with the data conditioned by 'mask'
#     sns.swarmplot(
#         x=scmat[mask][col],
#         hue=subinfo['r01x_bi'][mask],
#         palette={pos_cond: 'red', neg_cond: 'blue'}, 
#         s=3,
#         ax=ax  # Plot in the respective subplot
#     )
#     ax.set_title(col)  # Set title to the current column name

# # Adjust the layout and display the plots
# plt.tight_layout()
# plt.show()

# %%
