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
import anndata

PASCode.random_seed.set_seed(0)

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'
contrasts = ['c02x', 'r01x', 'c90x', 'c91x', 'c92x']
pos_conds = ['AD', 6, 'Sleep_WeightGain_Guilt_Suicide', 'WeightLoss_PMA', 'Depression_Mood']
subid_col = 'SubID'

#%% ######################################################################
#  prep
# ######################################################################
# print ('reading data...')
# meta = pd.read_csv('/home/che82/data/psychAD/MSSM_meta_obs-001.csv',index_col=0)
# suball = pd.read_csv('/home/che82/data/psychAD/metadata.csv',index_col=0)
# adata = anndata.AnnData(X=None, obs=meta)
# sub = ['M95030']  # this one is not in the contrast list?
# adata = adata[~adata.obs[subid_col].isin(sub),:]
# cellinfo = suball.loc[adata.obs[subid_col], :]
# cellinfo.index = adata.obs.index
# allmeta = pd.concat([adata.obs, cellinfo],axis=1)
# adata.obs = allmeta
# mask = ((~adata.obs['c02x'].isna()) | (~adata.obs['c90x'].isna()) | (~adata.obs['c91x'].isna()) | (~adata.obs['c92x'].isna()))
# adata = adata[mask]
# adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]
# adata.obs = adata.obs[[subid_col, 'c02x', 'r01x', 'c28x', 'c90x', 'c91x', 'c92x', 'Sex', 'Age', 'subclass', 'class', 'subtype']]
# adata.write_h5ad(DATA_PATH + "PsychAD/adata_AD_sel_NPS.h5ad")

# load data
adata = sc.read_h5ad(DATA_PATH + "PsychAD/adata_AD_sel_NPS_gxp.h5ad")
# ## add gxp
# adata_gxp = sc.read_h5ad('/media/che82/hechenfon/pead_freeze25/datasets2.5_M_selgenes2.h5ad')
# ind = [adata_gxp.obs.index.get_loc(i) for i in adata.obs.index]
# adata = anndata.AnnData(X=adata_gxp.X[ind, :], 
#                         obs=adata.obs, 
#                         var=adata_gxp.var, 
#                         uns=adata.uns, 
#                         obsm=adata.obsm, 
#                         obsp=adata.obsp)
adatas = []
for i in range(len(contrasts)):
    adatas.append(sc.read_h5ad(DATA_PATH + f"PsychAD/{contrasts[i][:-1]}_only_obs_obsm.h5ad"))
    adata.obs.loc[adatas[-1].obs.index, f'{contrasts[i][:-1]}_pac_score'] = adatas[-1].obs['pac_score'].values
    adata.obs[f'{contrasts[i][:-1]}_PAC+'] = adata.obs[f'{contrasts[i][:-1]}_pac_score'] > 0.5

# read PASCode color palette
class_palette_df = pd.read_csv("../class_palette.csv", index_col=0)
class_palette = dict(zip(class_palette_df.index, class_palette_df['color_hex']))

# get the ordered celltypes
print(adata.obs.groupby(['subclass', 'class']).size().unstack())
classes = ['Astro', 'Immune', 'Oligo', 'OPC', 'EN', 'IN', 'Endo', 'Mural']
subclasses = ['Astro', 'Micro', 'Immune', 'PVM', 'Oligo', 'OPC', 'EN_L2_3_IT',
 'EN_L3_5_IT_1', 'EN_L3_5_IT_2', 'EN_L3_5_IT_3', 'EN_L5_6_NP',
 'EN_L5_ET', 'EN_L6B', 'EN_L6_CT', 'EN_L6_IT_1', 'EN_L6_IT_2',
 'EN_NF',  # NOTE removed!
 'IN_ADARB2', 'IN_LAMP5_LHX6', 'IN_LAMP5_RELN', 'IN_PVALB',
 'IN_PVALB_CHC', 'IN_SST', 'IN_VIP', 'Endo', 'PC', 'SMC', 'VLMC']
celltypes = adata.obs.groupby(['subclass', 'class']).size().unstack().loc[subclasses, classes]
# print(celltypes)
celltypes.to_csv("class_subclass_cellnumber.csv")
celltypes.to_csv("celltypes_order.csv")
class_subclass_map = dict(zip(celltypes.columns, [celltypes.index[np.where(celltypes[col] > 0)[0]] for col in celltypes.columns]))

subclass_palette = pd.read_csv("../subclass_palette.csv", index_col=0)
subclass_int_map = dict(zip(subclass_palette.index, 1 + np.arange(len(subclass_palette))))

#%% ######################################################################
#  the PAC proportion plot
# ######################################################################
def pac_prop_df_helper(obs):
    # PAC proportion by subclass
    obs['PAC'] = PASCode.pac.assign_pac(
        scores = obs['pac_score'].values,
        mode='cutoff', cutoff=0.5
    )
    df0 = obs.groupby(['subclass', 'PAC']).size().unstack()
    df = df0.div(df0.sum(axis=1), axis='index').copy()
    df = df.drop(columns=[0])
    df = df.rename(columns={-1: 'PAC-', 1: 'PAC+'})
    # df = df.sort_values(by='PAC+', ascending=False)
    df0 = df0.loc[df.index]
    return df, df0

masks_dic = {
    'c02x': 'c02_100v100_mask',
    'c90x': 'c90_63v63_mask',
    'c91x': 'c91_30v30_mask',
    'c92x': 'c92_100v100_mask'
}

dfs = []
df0s = []
for i, contrast in enumerate(contrasts):
    d = adatas[i]
    mask = list(masks_dic.values())[i]
    res = pac_prop_df_helper(d.obs.loc[d.obs[mask]]) # NOTE balancing donors across conditions
    dfs.append(res[0].loc[subclasses])
    df0s.append(res[1].loc[subclasses])

for i in range(len(df0s)):
    df0s[i].rename(columns={-1: 'PAC-', 1: 'PAC+', 0.0:'Non-PAC'}, inplace=True)
    df0s[i].to_csv(f'pac_num_forsubclasses_for_{contrasts[i]}.csv')

#%% plot the color bars
titles = [
    'log2(AD PAC+ number)',
    'log2(NPS PAC+ number)',
    'log2(AD PAC- number)',
    'log2(NPS PAC- number)'
]
end_colors = [
    '#591496',
    '#0e38c2',
    '#1f7a0f',
    '#addeb3'
]
max_values = [
    df0s[0][1].max(),
    max(df0s[0][1].max(),df0s[1][1].max(),df0s[2][1].max(),df0s[2][1].max()),
    df0s[0][-1].max(),
    max(df0s[0][-1].max(),df0s[1][-1].max(),df0s[2][-1].max(),df0s[2][-1].max())
]
log2_max_values = [np.log2(x) for x in max_values]
for i in range(len(titles)):
    PASCode.utils.plot_color_bar(
        title=titles[i],
        vmin = 0,
        vmax = log2_max_values[i],
        st_color='white',
        end_color=end_colors[i],
        fontsize=30,
        save_path=f'pac_number_colorbar_{titles[i][:-7]}.pdf'
    )

#%% for bar color depth mapping 
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_hex
color_bars = []
for end_color, max_value in zip(end_colors, log2_max_values):
    color_bars.append(LinearSegmentedColormap.from_list('CustomColorBar', ['white', end_color]))
norms = [Normalize(vmin=0, vmax=max_val) for max_val in log2_max_values]
scalar_mappables = [ScalarMappable(norm=norm, cmap=color_bar) 
                    for norm, color_bar in zip(norms, color_bars)]
def get_bar_color(pac_num, scalar_mappable):
    return to_hex(scalar_mappable.to_rgba(pac_num))

#%% barplots with pac fractions and the 
rows, cols = len(contrasts), len(subclasses)
fig, axs = plt.subplots(rows, cols, 
                        figsize=(40, 20), 
                        sharex='col',
                        sharey='row') # NOTE
for i in range(rows):
    df = dfs[i]
    df0 = df0s[i].drop(columns=[0])
    for j in range(cols):
        pac_frac = df.iloc[j].values
        pac_num = np.log2(df0.iloc[j].values)

        scalar_mappable_pos = scalar_mappables[0] if i == 0 else scalar_mappables[1]
        scalar_mappable_neg = scalar_mappables[2] if i == 0 else scalar_mappables[3]
        color_pos = get_bar_color(pac_num[1], scalar_mappable_pos)
        color_neg = get_bar_color(pac_num[0], scalar_mappable_neg)

        axs[i, j].bar(
            ['PAC-', 'PAC+'], 
            pac_frac, 
            # color=['#1f7a0f', '#591496'] if i == 0 else ['#9e471b', '#0e38c2'] # V0
            color=[color_neg, color_pos] # V1
        )
        axs[i, j].set_xticklabels([])
        axs[i, j].grid(axis='y', linestyle='--', color='grey', alpha=0.9)
        if j == 0:
            if i == 0:
                axs[i, j].set_ylim([0, 0.35])
                axs[i, j].set_yticks([0, 0.1, 0.2, 0.3])
                axs[i, j].set_yticklabels(
                    labels=[0,0.1, 0.2, 0.3], 
                    fontsize=45)
            if i == 1:
                axs[i, j].set_ylim([0, 0.42])
                axs[i, j].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
                axs[i, j].set_yticklabels(
                    labels=[0,0.1, 0.2, 0.3, 0.4], 
                    fontsize=45)
            if i == 2:
                axs[i, j].set_ylim([0, 0.85])
                axs[i, j].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
                axs[i, j].set_yticklabels(
                    labels=[0, 0.2, 0.4, 0.6, 0.8], 
                    fontsize=45)
            if i == 3:
                axs[i, j].set_ylim([0, 0.15])
                axs[i, j].set_yticks([0, 0.05, 0.1, 0.15])
                axs[i, j].set_yticklabels(
                    labels=[0, .05, 0.1, .15], 
                    fontsize=45)
            axs[i, j].set_ylabel(contrasts[i], fontsize=50, rotation=90)
        if i == rows - 1: 
            axs[i, j].set_xlabel(df.index[j], fontsize=50, rotation=90)

plt.tight_layout()
plt.savefig('pac_prop_by_phenotype_celltype.pdf', bbox_inches='tight', dpi=600)
plt.show()


#%%
#######################################################################
# the feature importance by SHAP value plot
#######################################################################
from sklearn.ensemble import RandomForestClassifier
import shap

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'
contrasts = ['c02x', 'r01x', 'c90x', 'c91x', 'c92x']
pos_conds = ['AD', 6, 'Sleep_WeightGain_Guilt_Suicide', 'WeightLoss_PMA', 'Depression_Mood']

adatas = []
for i in range(len(contrasts)):
    adatas.append(sc.read_h5ad(DATA_PATH + f"PsychAD/{contrasts[i][:-1]}_only_obs_obsm.h5ad"))

#%% temp
scmat5_subtype = PASCode.utils.scmatrix(
    obses[-1], 
    subid_col,
    class_col='subtype',
    score_col='pac_score',
    column_order=None)
scmat5_subtype.to_csv("./scmat5_subtype.csv")

#%%
subclasses = ['Astro', 'Micro', 'Immune', 'PVM', 'Oligo', 'OPC', 'EN_L2_3_IT',
 'EN_L3_5_IT_1', 'EN_L3_5_IT_2', 'EN_L3_5_IT_3', 'EN_L5_6_NP',
 'EN_L5_ET', 'EN_L6B', 'EN_L6_CT', 'EN_L6_IT_1', 'EN_L6_IT_2',
 'EN_NF', 'IN_ADARB2', 'IN_LAMP5_LHX6', 'IN_LAMP5_RELN', 'IN_PVALB',
 'IN_PVALB_CHC', 'IN_SST', 'IN_VIP', 'Endo', 'PC', 'SMC', 'VLMC']

obses = [adatas[i].obs for i in range(len(adatas))]
trn_masks = ['c02_100v100_mask', 'r01_30v30_mask', 'c90_63v63_mask', 'c91_30v30_mask', 'c92_100v100_mask']

subid_col = 'SubID'
neg_cond = 'Control'
class_col = 'subclass'
for i, contrast in enumerate(contrasts):
    cond_col = f'{contrast}'
    pos_cond = pos_conds[i]
    trn_mask = trn_masks[i]
    obs = obses[i]

    exec(f"""
scmat{i+1} = PASCode.utils.scmatrix(
    obs, 
    subid_col,
    class_col=class_col,
    score_col='pac_score',
    column_order=subclasses)
scmat{i+1}.to_csv(f"scmat{i+1}.csv")
subinfo{i+1} = PASCode.utils.subject_info(
    obs,
    subid_col,
    columns=[cond_col])
scmat{i+1} = scmat{i+1}.loc[subinfo{i+1}.index]

scmat_trn{i+1} = PASCode.utils.scmatrix(
    obs[obs['{trn_mask}']],
    subid_col,
    class_col=class_col,
    score_col='pac_score',
    column_order=subclasses)
subinfo_trn{i+1} = PASCode.utils.subject_info(
    obs[obs['{trn_mask}']],
    subid_col,
    columns=[cond_col])
scmat_trn{i+1} = scmat_trn{i+1}.loc[subinfo_trn{i+1}.index]
    """)

#%%
num_repeat = 100 # NOTE
n_estimators = 100 # NOTE

feature_importances = []
shap_values_matrices = []

for i in range(len(pos_conds)):
    scmat = eval(f'scmat{i+1}')
    subinfo = eval(f'subinfo{i+1}')
    scmat_trn = eval(f'scmat_trn{i+1}')
    subinfo_trn = eval(f'subinfo_trn{i+1}')
    cond_col, pos_cond, neg_cond = contrasts[i], pos_conds[i], 'Control'
    if pos_cond == 6: neg_cond = 0

    shap_values = np.zeros_like(scmat.values)
    subinfo['subject_score'] = 0.0
    for seed in np.arange(num_repeat):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X=scmat_trn, 
                y=subinfo_trn[cond_col].map({pos_cond: 1, neg_cond: 0}))
        explainer = shap.TreeExplainer(clf)
        
        shap_values += explainer.shap_values(scmat.values)[1]
        subinfo['subject_score'] += clf.predict_proba(scmat)[:, 1]

    subinfo['subject_score'] /= num_repeat
    shap_values /= num_repeat
    shap_values = pd.DataFrame(shap_values, index=scmat.index, columns=scmat.columns)
    feature_importance = np.abs(shap_values).mean().sort_values(ascending=False)    
    # feature_importance /= feature_importance.max()

    shap_values.to_csv(f"shap_values_{cond_col}.csv")

    shap_values_matrices.append(shap_values)
    feature_importances.append(feature_importance)

#%% save


#%%
### option 1
df = pd.DataFrame(index=contrasts, columns=subclasses)
for i in range(len(contrasts)):
    df.iloc[i] = feature_importances[i]
df.to_csv("feature_importance_by_shap.csv")
# ### option 2
# df = pd.DataFrame(index=contrasts, columns=subclasses)
# for i in range(len(contrasts)):
#     d = adatas[i] # donor balanced
#     mask = d.obs[trn_masks[i]]
#     res = d.obs[mask].groupby('subclass')['pac_score'].mean()
#     df.loc[contrasts[i], res.index] = res.values
# df.to_csv("feature_importance_option2.csv")
# ### option 3
# df = pd.DataFrame(index=contrasts, columns=subclasses)
# for i in range(len(contrasts)):
#     res = eval(f"scmat_trn{i+1}").mean()
#     df.loc[contrasts[i], res.index] = res.values
# df.to_csv("feature_importance_option3.csv")
# ### option 4
# df = pd.DataFrame(index=contrasts, columns=subclasses)
# for i in range(len(contrasts)):
#     res = shap_values_matrices[i].mean()
#     df.loc[contrasts[i], res.index] = res.values
# df.to_csv("feature_importance_option4.csv")

#%% barplot
mat = pd.read_csv("feature_importance_by_shap.csv", index_col=0)
colors = ['#591496', '#0e38c2', '#0e38c2', '#0e38c2']
rows = mat.shape[0]
fig, axs = plt.subplots(rows, 1, figsize=(40, 45))
for i in range(rows):
    sorted_row = mat.iloc[i, :].sort_values(ascending=False)
    sns.barplot(x=[str(subclass_int_map[sorted_row.index[i]]) for i in range(len(sorted_row.index))], 
                y=sorted_row.values, 
                ax=axs[i], 
                palette=[colors[i]])
    if i == 0:
        axs[i].set_title(f'Cell-type importance from SHAP values', fontsize=60)
    axs[i].grid(axis='y', linestyle='--', color='grey', alpha=0.9)
    axs[i].tick_params(axis='x', labelsize=45)
    axs[i].tick_params(axis='y', labelsize=45)
plt.tight_layout()
plt.savefig('celltype_importance.pdf', bbox_inches='tight', dpi=600)
plt.show()
#%% ######################################################################
# DEG number heatmap plot
#######################################################################
mat_up = pd.DataFrame(index=contrasts, columns=subclasses)
mat_down = pd.DataFrame(index=contrasts, columns=subclasses)
for i in range(4):
    deg_num = pd.read_csv("/home/che82/athan/PASCode/code/github_repo/figures/fig6/" + contrasts[i] + "_pval_0.05_FC_0_5/" + contrasts[i] + "_num_DEX_genes.csv", index_col=0)
    mat_up.loc[contrasts[i], deg_num.index] = deg_num['Upregulated'].values
    mat_down.loc[contrasts[i], deg_num.index] = deg_num['Downregulated'].values
mat_up.loc[:, subclasses].to_csv("deg_up_num.csv")
mat_up.to_csv("deg_up_num.csv")
mat_down.loc[:, subclasses].to_csv("deg_down_num.csv")
mat_down.to_csv("deg_down_num.csv")
mat_up.fillna(0, inplace=True)
mat_down.fillna(0, inplace=True)

#%%
#######################################################################d
# DEG circle plot
#######################################################################
import glob

dics_up = []
dics_down = []
for contrast_idx in range(4):
    data_paths = sorted(glob.glob(f'/home/che82/athan/PASCode/code/github_repo/figures/fig6/{contrasts[contrast_idx]}_pval_0.05_FC_0_5/DEX_pac_filtered*'))
    dic_up = {}
    dic_down = {}
    for i in range(len(data_paths)):
        deg = pd.read_csv(data_paths[i], index_col=0)
        if deg.shape[0] == 0:
            print('No data! - ', i)
            raise ValueError
        assert deg['pvals_adj'].max() <= 0.05
        for ctp in subclasses:
            if ctp in data_paths[i]:
                break
        dic_up[ctp] = list(deg['names'].values[deg['logfoldchanges'] > 0])
        dic_down[ctp] = list(deg['names'].values[deg['logfoldchanges'] < 0])

    dics_up.append(dic_up)
    dics_down.append(dic_down)

# # #%% taking common ctp
# # for i in range(4):
# #     for j in range(i, 4):
# #         up_down = ['up', 'down']
# #         for k in range(len(up_down)):
# #             dic1 = dics_up[i] if k == 0 else dics_down[i]
# #             dic2 = dics_up[j] if k == 0 else dics_down[j]
# #             ctps1 = list(dic1.keys())
# #             ctps2 = list(dic2.keys())
# #             common_ctps = list(set(ctps1).intersection(ctps2))
# #             mat = pd.DataFrame(index=common_ctps, columns=common_ctps)
# #             for ctp1 in common_ctps:
# #                 for ctp2 in common_ctps:
# #                     mat.loc[ctp1, ctp2] = len(np.intersect1d(dic1[ctp1], dic2[ctp2]))
# #             mat.to_csv(f"deg_circle_{contrasts[i]}_{contrasts[j]}_{up_down[k]}_common_ctp.csv")

# #             # ### number of non-overlapping upgualted genes
# #             # mat = pd.DataFrame(index=ctps, columns=ctps)
# #             # for ctp1 in ctps:
# #             #     for ctp2 in ctps:
# #             #         mat.loc[ctp1, ctp2] = len(np.union1d(dic_up[ctp1], dic_up[ctp2])) - len(np.intersect1d(dic_up[ctp1], dic_up[ctp2]))
# #             # mat.to_csv("deg_circle_c02_up_nonovlp.csv")


# #%% not taking commonm ctp
# for i in range(4):
#     for j in range(i, 4):
#         up_down = ['up', 'down']
#         for k in range(len(up_down)):
#             dic1 = dics_up[i] if k == 0 else dics_down[i]
#             dic2 = dics_up[j] if k == 0 else dics_down[j]
#             ctps1 = list(dic1.keys())
#             ctps2 = list(dic2.keys())
#             mat = pd.DataFrame(index=ctps1, columns=ctps2)
#             for ctp1 in ctps1:
#                 for ctp2 in ctps2:
#                     mat.loc[ctp1, ctp2] = len(np.intersect1d(dic1[ctp1], dic2[ctp2]))
#             mat.to_csv(f"deg_circle_{contrasts[i]}_{contrasts[j]}_{up_down[k]}.csv")

#             # ### number of non-overlapping upgualted genes
#             # mat = pd.DataFrame(index=ctps, columns=ctps)
#             # for ctp1 in ctps:
#             #     for ctp2 in ctps:
#             #         mat.loc[ctp1, ctp2] = len(np.union1d(dic_up[ctp1], dic_up[ctp2])) - len(np.intersect1d(dic_up[ctp1], dic_up[ctp2]))
#             # mat.to_csv("deg_circle_c02_up_nonovlp.csv")

# #%% plot color bars
# colors_up = [
#   '#ffffff',
#   '#ffffe6',
#   '#fff4db',
#   '#ffe9d0',
#   '#ffdebf',
#   '#ffd4a9',
#   '#ffc994',
#   '#ffbe7e',
#   '#ffb369',
#   '#ffa953',
#   '#ff9e3e',
#   '#ff9428',
#   '#ff8913',
#   '#ff7f00',
#   '#ff7400',
#   '#ff6900',
#   '#ff5600',
#   '#ff4000',
#   '#ff2b00',
#   '#ff1500',
#   '#ff0000'
# ]
# colors_down = ['#ffffff', 
#             '#f2f2ff', 
#             '#e5e5ff', 
#             '#d8d8ff', 
#             '#ccccff', 
#             '#bfbfff', 
#             '#b2b2ff', 
#             '#a5a5ff', 
#             '#9999ff', 
#             '#8c8cff', 
#             '#7f7fff', 
#             '#7272ff', 
#             '#6666ff', 
#             '#5959ff', 
#             '#4c4cff', 
#             '#3f3fff', 
#             '#3333ff', 
#             '#2626ff', 
#             '#1919ff', 
#             '#0c0cff', 
#             '#0000ff']

# for i in range(4):
#     for j in range(i, 4):
#         up_down = ['up', 'down']
#         for k in up_down:
#             mat = pd.read_csv(f"deg_circle_{contrasts[i]}_{contrasts[j]}_{k}.csv", index_col=0)
#             vmin = mat.min().min()
#             vmax = mat.max().max()
#             print('contrast: ', contrasts[i], contrasts[j], k, vmin, vmax)
#             PASCode.utils.plot_color_bar(
#                 title=f"Number of overlapped DEGs",
#                 vmin = vmin,
#                 vmax = vmax,
#                 colors=colors_up if k == 'up' else colors_down,
#                 fontsize=30,
#                 save_path=f'deg_circle_{contrasts[i]}_{contrasts[j]}_colorbar_{k}.pdf'
#             )


#%%
###############################################################################
# UpsetR to show AD-NPS common DEG numbers ... celltypes?
###############################################################################
# for i in range(1, 4):
#     dics_up[i] = dict(zip(
#         [(key + '_') for key in list(dics_up[3].keys())], 
#         dics_up[3].values()
#     ))

import json
for i in range(len(contrasts)):
    with open(f'dic_up_{contrasts[i]}.json', 'w') as json_file:
        json.dump(dics_up[i], json_file)

#%%
###############################################################################
# for dotplot
###############################################################################
import json
with open(f'dic_up_c02x.json', 'r') as json_file:
    dic1 = json.load(json_file)
with open(f'dic_up_c90x.json', 'r') as json_file:
    dic2 = json.load(json_file)
with open(f'dic_up_c91x.json', 'r') as json_file:
    dic3 = json.load(json_file)
with open(f'dic_up_c92x.json', 'r') as json_file:
    dic4 = json.load(json_file)

# gset1 = np.intersect1d(dic1['Oligo'], dic2['Oligo'])
# gset2 = np.intersect1d(dic1['Oligo'], dic4['Oligo'])
# gset3 = np.intersect1d(dic3['Oligo'], dic1['Oligo'])
# gset4 = np.intersect1d(np.intersect1d(dic2['Oligo'], dic3['Oligo']), dic4['Oligo'])
# gset5 = np.intersect1d(np.intersect1d(dic2['Oligo'], dic3['Oligo']), np.intersect1d(dic4['Oligo'], dic1['Oligo']))
# gset = np.union1d(np.union1d(gset1, gset2), np.union1d(gset3, gset4))
# gset = np.union1d(np.union1d(gset1, gset2), gset3)
# gseti = np.intersect1d(
#     gset,
#     adata.var.index
# )
# gset1 = dic1['Oligo']
# gset2 = dic2['Oligo']
# gset3 = dic3['Oligo']
# gset4 = dic4['Oligo']
# np.setdiff1d(gset1, np.)

missing_g = np.setdiff1d(gset, gseti) # TODO
all_gxp = sc.read_h5ad(DATA_PATH + 'PsychAD/c02x-all.protein_coding.h5ad')
adata = adata[np.intersect1d(adata.obs.index, all_gxp.obs.index), :]
adata = adata[all_gxp.obs.index, :]
new_X = np.hstack([
    adata.X.toarray(),
    all_gxp.X[:, [np.where(all_gxp.var.index==missing_g[i])[0][0] for i in range(len(missing_g))]].toarray()
])
new_var = pd.concat([
    adata.var,
    pd.DataFrame(index=all_gxp.var.loc[missing_g].index)
])
adata = anndata.AnnData(
    X = new_X,
    obs=adata.obs,
    var=new_var
)

#%%
ep1s = []
perc_exp1s = []
ep2s = []
perc_exp2s = []
ep3s = []
perc_exp3s = []
ep4s = []
perc_exp4s = []
for i in range(len(gset)):
    # # has keyerror since all_gxp miss some donors
    # ep1 = all_gxp[
    #     adata.obs.index[adata.obs['c02_PAC+']], 
    #     np.where(all_gxp.var.index==gset[i])[0][0]].X.toarray().flatten()
    # ep2 = all_gxp[
    #     adata.obs.index[adata.obs['c90_PAC+']], 
    #     np.where(all_gxp.var.index==gset[i])[0][0]].X.toarray().flatten()
    # ep3 = all_gxp[
    #     adata.obs.index[adata.obs['c91_PAC+']], 
    #     np.where(all_gxp.var.index==gset[i])[0][0]].X.toarray().flatten()
    # ep4 = all_gxp[
    #     adata.obs.index[adata.obs['c92_PAC+']], 
    #     np.where(all_gxp.var.index==gset[i])[0][0]].X.toarray().flatten()

    # has keyerror since all_gxp miss some donors
    ep1 = adata[
        adata.obs.index[adata.obs['c02_PAC+']], 
        np.where(adata.var.index==gset[i])[0][0]].X.toarray().flatten()
    ep2 = adata[
        adata.obs.index[adata.obs['c90_PAC+']], 
        np.where(adata.var.index==gset[i])[0][0]].X.toarray().flatten()
    ep3 = adata[
        adata.obs.index[adata.obs['c91_PAC+']], 
        np.where(adata.var.index==gset[i])[0][0]].X.toarray().flatten()
    ep4 = adata[
        adata.obs.index[adata.obs['c92_PAC+']], 
        np.where(adata.var.index==gset[i])[0][0]].X.toarray().flatten()

    perc_exp1 = (ep1>0).sum() / len(ep1)
    perc_exp2 = (ep2>0).sum() / len(ep2)
    perc_exp3 = (ep3>0).sum() / len(ep3)
    perc_exp4 = (ep4>0).sum() / len(ep4)

    # total_len = len(ep1) + len(ep2) + len(ep3) + len(ep4)
    # perc_exp1 = (ep1>0).sum() / total_len
    # perc_exp2 = (ep2>0).sum() / total_len
    # perc_exp3 = (ep3>0).sum() / total_len
    # perc_exp4 = (ep4>0).sum() / total_len

    avg_exp1 = ep1.mean()
    avg_exp2 = ep2.mean()
    avg_exp3 = ep3.mean()
    avg_exp4 = ep4.mean()

    ep1s.append(avg_exp1)
    perc_exp1s.append(perc_exp1)
    ep2s.append(avg_exp2)
    perc_exp2s.append(perc_exp2)
    ep3s.append(avg_exp3)
    perc_exp3s.append(perc_exp3)
    ep4s.append(avg_exp4)
    perc_exp4s.append(perc_exp4)

avg_exp = np.concatenate([ep1s, ep2s, ep3s, ep4s]).flatten()
pct_exp = np.concatenate([perc_exp1s, perc_exp2s, perc_exp3s, perc_exp4s]).flatten()
## save 
import csv
with open('gset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(gset)
with open('avg_exp.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(avg_exp)
with open('pct_exp.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(pct_exp)







# len(dic1['Oligo'])
# len(dic2['Oligo'])
# len(dic3['Oligo'])
# len(dic4['Oligo'])

# mask1 = adata.obs['c02x']=='AD'
# mask2 = adata.obs['c90x']=='Sleep_WeightGain_Guilt_Suicide'
# mask3 = adata.obs['c91x']=='WeightLoss_PMA'
# mask4 = adata.obs['c92x']=='Depression_Mood'

# (adata.obs['subclass']=='Oligo').sum()
# ((adata.obs['subclass']=='Oligo') & mask1).sum()
# ((adata.obs['subclass']=='Oligo') & mask2).sum()
# ((adata.obs['subclass']=='Oligo') & mask3).sum()
# ((adata.obs['subclass']=='Oligo') & mask4).sum()
# ((adata.obs['subclass']=='Oligo') & mask1 & mask2).sum()
# ((adata.obs['subclass']=='Oligo') & mask1 & mask3).sum()
# ((adata.obs['subclass']=='Oligo') & mask1 & mask4).sum()
# ((adata.obs['subclass']=='Oligo') & mask3 & mask2).sum()
# ((adata.obs['subclass']=='Oligo') & mask4 & mask2).sum()
# ((adata.obs['subclass']=='Oligo') & mask1 & mask2 & mask3 & mask4).sum()





#%% temp
import scanpy as sc
adata = sc.read_h5ad("/media/che82/hechenfon/pead_freeze25/datasets2.5_M_selgenes2.h5ad")

adata





