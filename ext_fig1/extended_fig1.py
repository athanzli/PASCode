#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('../..')
import PASCode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rand_seed = 0
PASCode.random_seed.set_seed(rand_seed)

import seaborn as sns
import scanpy as sc
import anndata

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/'
subid_col = 'SubID'

#%%
###############################################################################
# load data
###############################################################################
print ('reading data...')
meta = pd.read_csv('/home/che82/data/psychAD/MSSM_meta_obs-001.csv',index_col=0)
suball = pd.read_csv('/home/che82/data/psychAD/metadata.csv',index_col=0)
adata = anndata.AnnData(X=None, obs=meta)
sub = ['M95030']  # this one is not in the contrast list?
adata = adata[~adata.obs[subid_col].isin(sub),:]
cellinfo = suball.loc[adata.obs[subid_col], :]
cellinfo.index = adata.obs.index
allmeta = pd.concat([adata.obs, cellinfo],axis=1)
adata.obs = allmeta
mask = ((~adata.obs['c02x'].isna()) | (~adata.obs['c90x'].isna()) | (~adata.obs['c91x'].isna()) | (~adata.obs['c92x'].isna())) | (~adata.obs['c28x'].isna()) | (~adata.obs['r01x'].isna())
adata = adata[mask]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]
adata.obs = adata.obs[[
    subid_col, 'c02x', 'r01x', 'c28x', 'c90x', 'c91x', 'c92x', 
    'Sex', 'Age', 'Ethnicity', 'subclass', 'class', 'subtype']]

#%%
###############################################################################
# donor stats for the four phenotypes HAS BUGS
###############################################################################
dinfo = PASCode.subject_info(
    adata.obs,
    subid_col,
    columns=['c02x', 'r01x', 'c90x', 'c91x', 'c92x', 'Sex', 'Age', 'Ethnicity', 'c28x']
)
dinfo['AD-resilience'] = dinfo['c28x']
print(dinfo['c28x'].value_counts())
dinfo['Braak'] = dinfo['r01x']
dinfo = dinfo.drop(columns=['cell_num'])

dinfo.fillna('Na', inplace=True)

# dinfo = pd.read_csv("../fig6/dinfo_for_fig6.csv", index_col=0)
# dinfo['Ethnicity_original'] = dinfo['Ethnicity']
# mask = (dinfo['Ethnicity'] == 'EAS') | (dinfo['Ethnicity'] == 'SAS')
# dinfo.loc[mask, 'Ethnicity'] = 'AS'
# dinfo.to_csv('dinfo_for_extended_fig1.csv')
# dinfo.to_csv('../fig6/dinfo_for_fig6.csv')
## dinfo = pd.read_csv("./dinfo_for_extended_fig1.csv", index_col=0)

# dinfo['c90x'] = dinfo['c90x'].map({'Control': 'GltSuic_CTL', 'Sleep_WeightGain_Guilt_Suicide':'GltSuic'})
# dinfo['c91x'] = dinfo['c91x'].map({'Control': 'WtLoss_CTL', 'WeightLoss_PMA':'WtLoss'})
# dinfo['c92x'] = dinfo['c92x'].map({'Control': 'Dep_CTL', 'Depression_Mood':'Dep'})

# dinfo.groupby(['c90x', 'c91x', 'c92x']).size()
# dinfo.groupby(['c90x', 'c91x']).size()
# dinfo.groupby(['c90x', 'c92x']).size()
# dinfo.groupby(['c91x', 'c92x']).size()

# order color bar
dinfo['ZBraak'] = dinfo['Braak'].map({
    'Na':0.0,
    0.0:1.0,
    1.0:2.0,
    2.0:3.0,
    3.0:4.0,
    4.0:5.0,
    5.0:6.0,
    6.0:7.0,
})
dinfo['Zc02x']=dinfo['c02x'].replace('AD','ZAD')
dinfo['Zc02x']=dinfo['Zc02x'].replace('Control','YControl')
dinfo['Zc90x']=dinfo['c90x'].replace('Control','YControl')
dinfo['Zc90x']=dinfo['Zc90x'].replace('Sleep_WeightGain_Guilt_Suicide','ZSleep_WeightGain_Guilt_Suicide')
dinfo['Zc91x']=dinfo['c91x'].replace('Control','YControl')
dinfo['Zc91x']=dinfo['Zc91x'].replace('WeightLoss_PMA','ZWeightLoss_PMA')
dinfo['Zc92x']=dinfo['c92x'].replace('Control','YControl')
dinfo['Zc92x']=dinfo['Zc92x'].replace('Depression_Mood','ZDepression_Mood')
dinfo['ZAD-resilience']=dinfo['AD-resilience'].replace('Control','YControl')
dinfo=dinfo.sort_values(
    by=['ZBraak','Zc02x','ZAD-resilience','Zc92x','Zc90x','Zc91x','Age','Sex','Ethnicity']
)

dinfo.to_csv('dinfo_for_extended_fig1.csv')
# dinfo = pd.read_csv('dinfo_for_extended_fig1.csv', index_col=0)

#%%
###############################################################################
################################## 
###############################################################################
# Na_color = '#dadada'
# Na_color = '#e6e6e6'
# color_dic = {
#     "AD": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
#     "c02x": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
#     "SleepWeightGainGuiltSuicide": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "c90x": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "WeightLossPMA": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "c91x": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "DepressionMood": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},
#     "c92x": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},
#     "Sex": {'Male': '#1d63ad', 'Na': Na_color, 'Female': '#8bbcd0'},
#     "c28x": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
#     "AD_strict_and_Resilient": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
#     "Braak stage": {'0.0': "#1f7a0f",
#                     '1.0': "#389e26",
#                     '2.0': "#6dc95d",
#                     '3.0': "#7389d1",
#                     '4.0': "#4969d1",
#                     '5.0': "#9e61d4",
#                     '6.0': "#591496",
#                     'Na': Na_color},
#     "Braak stage": {0: "#1f7a0f",
#                     1: "#389e26",
#                     2: "#6dc95d",
#                     3: "#7389d1",
#                     4: "#4969d1",
#                     5: "#9e61d4",
#                     6: "#591496",
#                     'Na': Na_color},
#     "Braak": {'0.0': "#1f7a0f",
#                     '1.0': "#389e26",
#                     '2.0': "#6dc95d",
#                     '3.0': "#7389d1",
#                     '4.0': "#4969d1",
#                     '5.0': "#9e61d4",
#                     '6.0': "#591496",
#                     'Na': Na_color},
#     "Braak": {0.0: "#1f7a0f",
#               1.0: "#389e26",
#               2.0: "#6dc95d",
#               3.0: "#7389d1",
#               4.0: "#4969d1",
#               5.0: "#9e61d4",
#               6.0: "#591496",
#               'Na': Na_color},
#     "Ethnicity": {
#         'EUR': '#1d63ac', # TODO 1d63ad
#         'AMR': '#fdb584',
#         'EAS': '#389e26',
#         'AFR': '#8bbcd1', # TODO 8bbcd0
#         'SAS': '#d4d19c',
#         'EAS_SAS': '#9dc449',
#         'Unknown': '#dadada'
#     },
#     'Brain bank': {
#         'MSSM': '#20b8da',
#         'HBCC': '#fdb584',
#         'RUSH': '#2AA66F' # https://www.rush.edu/
#     }
# }

# color_dic = {
#     "c02x": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
#     "c90x": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "c91x": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
#     "c92x": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},
#     "Sex": {'Male': '#1d63ad', 'Na': Na_color, 'Female': '#8bbcd0'},
#     "c28x": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
#     "Braak": {0: "#1f7a0f",
#               1: "#389e26",
#               2: "#6dc95d",
#               3: "#7389d1",
#               4: "#4969d1",
#               5: "#9e61d4",
#               6: "#591496",
#               'Na': Na_color},
#     "Ethnicity": {
#         'EUR': '#1d63ad',
#         'AMR': '#fdb584',
#         'EAS': '#389e26',
#         'AFR': '#8bbcd0',
#         'SAS': '#d4d19c'
#     }
# }

# version: 02/15/2024. using psychad official palette
palette = pd.read_csv("/home/che82/athan/PASCode/code/github_repo/figures/PsychAD_color_palette_230921.csv", index_col=0)

Na_color = '#e6e6e6'
nps_color = '#0e38c2'
nps_ctl_color = '#addeb3'
color_dic = {
    "AD": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
    "c02x": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
    "SleepWeightGainGuiltSuicide": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "c90x": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "WeightLossPMA": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "c91x": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "DepressionMood": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},
    "c92x": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},

    "c28x": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
    "AD_strict_and_Resilient": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
    "Braak stage": {'0.0': "#1f7a0f",
                    '1.0': "#389e26",
                    '2.0': "#6dc95d",
                    '3.0': "#7389d1",
                    '4.0': "#4969d1",
                    '5.0': "#9e61d4",
                    '6.0': "#591496",
                    'Na': Na_color},
    # "Braak stage": {0: "#1f7a0f",
    #                 1: "#389e26",
    #                 2: "#6dc95d",
    #                 3: "#7389d1",
    #                 4: "#4969d1",
    #                 5: "#9e61d4",
    #                 6: "#591496",
    #                 'Na': Na_color},
    "Braak": {'0.0': "#1f7a0f",
              '1.0': "#389e26",
              '2.0': "#6dc95d",
              '3.0': "#7389d1",
              '4.0': "#4969d1",
              '5.0': "#9e61d4",
              '6.0': "#591496",
              'Na': Na_color},
    # "Braak": {0.0: "#1f7a0f",
    #           1.0: "#389e26",
    #           2.0: "#6dc95d",
    #           3.0: "#7389d1",
    #           4.0: "#4969d1",
    #           5.0: "#9e61d4",
    #           6.0: "#591496",
    #           'Na': Na_color},

    "Sex": {'Male': '#40E0D0', 'Na': Na_color, 'Female': '#FF6B00'},
    "Ethnicity": {
        'EUR': '#D81B60',
        'AMR': '#1E88E5',
        'AFR': '#57A860',
        
        'EAS': '#004D40',
        'SAS': '#004D40',
        'EAS_SAS': '#004D40',
        'AS': '#004D40',
        
        'Unknown': '#FFC107'
    },
}

#%%
###############################################################################
# stacked version
###############################################################################
dinfo = pd.read_csv('dinfo_for_extended_fig1.csv', index_col=0)

phenotypes = ['AD', 'SleepWeightGainGuiltSuicide', 'WeightLossPMA', 'DepressionMood']
dinfo = dinfo.rename(columns={
    'c02x':'AD',
    'c90x':'SleepWeightGainGuiltSuicide',
    'c91x':'WeightLossPMA',
    'c92x':'DepressionMood',
    'Braak':'Braak stage'
})
# dinfo.fillna('Na', inplace=True)

all_cat_colors = {}
for category, value_color_pairs in color_dic.items():
    all_cat_colors.update(value_color_pairs)

#%%
for i in range(len(phenotypes)):
    df = dinfo[dinfo[phenotypes[i]]!='Na']
    print('Number of donors:', df.shape[0])
    df = df[[phenotypes[i], 'Braak stage', 'Sex', 'Ethnicity']]
    df_new = {col: df[col].value_counts(normalize=True) for col in df.columns} # NOTE choose normalize
    df_new_nonorm = {col: df[col].value_counts(normalize=False) for col in df.columns} # NOTE choose normalize
    df_new = pd.DataFrame(df_new).T.fillna(0)
    df_new_nonorm = pd.DataFrame(df_new_nonorm).T.fillna(0)

    colors = [all_cat_colors.get(col) for col in df_new.columns]
    if phenotypes[i] != 'AD':
        j = list(df_new.columns).index('Control')
        colors[j] = nps_ctl_color

    # df_new.plot(
    #     kind='bar',
    #     stacked=True,
    #     figsize=(6,6),
    #     color=colors,
    #     legend=False
    # )
        
    # df_new_nonorm.plot(
    #     kind='bar',
    #     stacked=True,
    #     figsize=(6,6),
    #     color=colors,
    #     legend=False
    # )

    df_new_nonorm = df_new_nonorm.astype(int)
    ax = df_new_nonorm.plot(kind='bar', stacked=True, figsize=(6,6), color=colors, legend=False)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if height > 0:  # or some small threshold instead of 0 to avoid clutter
            ax.annotate(f'{height}', (p.get_x() + width / 2, p.get_y() + height * 0.5), ha='center')

    # plt.ylim(0, 1)
    # plt.savefig(f"stacked_barplot_for_{phenotypes[i]}.pdf", dpi=600)
    plt.show()

    # legends = []
    # for category_name, color_map in color_dic.items():
    #     legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    #     legend = axes[i].legend(handles=legend_patches, title=category_name, loc='upper right')
    #     axes[i].add_artist(legend)
    #     legends.append(legend)
    # legends[-1].set_bbox_to_anchor((2, 1))


#%%
###############################################################################
#
###############################################################################
# ## non stacking version
# phenotypes = ['AD', 'SleepWeightGainGuiltSuicide', 'WeightLossPMA', 'DepressionMood']
# dinfo = dinfo.rename(columns={
#     'c02x':'AD',
#     'c90x':'SleepWeightGainGuiltSuicide',
#     'c91x':'WeightLossPMA',
#     'c92x':'DepressionMood',
#     'Braak':'Braak stage'
# })


# def map_color(row):
#     return color_dic[row['cat']][row['value']]

# # all_cat_colors = {}
# # for category, value_color_pairs in color_dic.items():
# #     all_cat_colors.update(value_color_pairs)

# for i in range(len(phenotypes)):
#     df = dinfo[dinfo[phenotypes[i]]!='Na']
#     df = df[[phenotypes[i], 'Braak stage', 'Sex', 'Ethnicity']]
#     df_melted = df.melt(var_name='cat', value_name='value')
#     counts = df_melted.groupby(['cat', 'value']).size().reset_index(name='counts')
#     counts['color'] = counts.apply(map_color, axis=1)
    
#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         x='value',
#         y='counts',
#         hue='cat', 
#         data=counts,
#         palette=counts.set_index('value')['color'])
#     # plt.title('Bar Plot for Multiple Categories')
#     plt.ylabel('Number of donors')
#     # plt.xlabel('Values')
#     plt.show()


# plt.savefig('./donor_info_summary.pdf', dpi=600)
# plt.show()


#%%
# # V0
# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 11))
# contrasts = ['c02x', 'c90x', 'c91x', 'c92x']

# sns.set_style("whitegrid")

# for i in range(len(contrasts)):
#     original_df = dinfo[dinfo[contrasts[i]]!='Na']
#     for j, col in enumerate([contrasts[i], 'Braak', 'Sex', 'Age']):
#         ax = axes[i, j]
#         df = original_df.copy()
#         df = df[df[col] != 'Na']

#         if col == 'Age':
#             # sns.set_context("talk")
#             sns.histplot(df, x=col, bins=50, 
#                          kde=True, color='royalblue', ax=ax)
#         else:
#             if col == 'Braak':
#                 df[col] = df[col].astype(float).astype(int)

#             values = df[col].value_counts().values
#             labels = df[col].value_counts().index
#             colors = [color_dic[col][x] for x in labels]

#             ax.pie(
#                 values, 
#                 labels=labels,
#                 autopct='%.1f%%',
#                 colors=colors,
#             )

# plt.tight_layout()
# plt.savefig('./donor_info_summary.pdf', dpi=600)
# plt.show()

# V0' for fig1 B 12/22/2023
dinfo = suball.copy()
dinfo['Braak'] = dinfo['r01x']
cols = ['c02x', 'c90x', 'c91x', 'c92x', 'Braak', 'c28x']

label_order_dic = {
    'c02x': [1,2,0],
    'c90x': [2,1,0],
    'c91x': [1,2,0],
    'c92x': [2,1,0],
    'Braak': [7,5,3,2,6,4,1,0],
    'c28x': [1,3,2,0],
}

for i, col in enumerate(cols):
    dinfo.loc[dinfo[col].isna(), col] = 'Na'                                                                                                                                                                      

    values = dinfo[col].value_counts().values
    labels = dinfo[col].value_counts().index
    idx = label_order_dic[col]
    values = values[idx]
    labels = labels[idx]
    colors = [color_dic[col][x] for x in labels]

    plt.pie(
        values, 
        labels=labels,
        autopct='%.1f%%',
        colors=colors,
    )
    plt.savefig(f"./pie_chart_for_{col}.pdf", dpi=600)
    plt.show()

# %% 
###########################################################################
##
###########################################################################
suball = pd.read_csv('/home/che82/data/psychAD/metadata.csv',index_col=0)
dinfo = suball[['Brain_bank', 'Sex', 'Ethnicity', 'c02x', 'c90x', 'c91x', 'c92x', 'c28x', 'r01x']]

dinfo = dinfo.rename(columns={
    'Brain_bank': 'Brain bank',
    'c02x':'AD',
    'r01x':'Braak',
    'c28x': 'AD-strict and AD-resilient',
    'c90x':'SleepWeightGainGuiltSuicide',
    'c91x':'WeightLossPMA',
    'c92x':'DepressionMood',
})
dinfo.columns
dinfo = dinfo[['Brain bank', 'Sex', 'Ethnicity', 'AD', 'Braak', 'AD-strict and AD-resilient', 'SleepWeightGainGuiltSuicide', 'WeightLossPMA', 'DepressionMood']]
# dinfo.fillna('Na', inplace=True)

all_cat_colors = {}
for category, value_color_pairs in color_dic.items():
    all_cat_colors.update(value_color_pairs)

df = dinfo.copy()                                                                                   
df_new = {col: df[col].value_counts(normalize=False) for col in df.columns} # TODO choose normalize
df_new = pd.DataFrame(df_new).T.fillna(0)
df_new.plot(
    kind='bar',
    stacked=True,                                                       
    figsize=(16,6),                                                                                                                                  
    color=[all_cat_colors.get(col) for col in df_new.columns],
    legend=False
)
# plt.ylim(0, 1)
plt.savefig(f"stacked_barplot_for_all_donors.pdf", dpi=600)
plt.show()

# %%
###########################################################################
##
###########################################################################
#%%
subclass_palette = pd.read_csv('../subclass_palette.csv', index_col=0)

#%%
# df = allmeta[['class','subclass','subtype']]
# df = df.groupby(['class','subclass']).size().unstack().fillna(0)
# df.plot(
#     kind='bar',
#     stacked=True, 
#     figsize=(16,6),                                                                                                                                  
#     color=[subclass_palette['color_hex'].loc[col] for col in df.columns],
#     legend=False
# )
# plt.savefig(f"stacked_barplot_for_celltype.pdf", dpi=600)
# plt.show()


subtype_palette

df = allmeta[['class','subclass','subtype']]
df = df.groupby(['subclass','subtype']).size().unstack().fillna(0)
lognum = np.log2(df.sum(1) + 1)
frac = df.div(df.sum(axis=1), axis=0)
assert (frac.index == lognum.index).all()
trans = frac.mul(lognum, axis=0)
trans.plot(
    kind='bar',
    stacked=True, 
    figsize=(16,6),                                                                                                                                  
    color=[subtype_palette['color_hex'].loc[col] for col in df.columns],
    legend=False
)
plt.savefig(f"stacked_barplot_for_celltype.pdf", dpi=600)
plt.show()


df.columns

"""
'Astro_ADAMTSL3', 'Astro_GRIA1', 'Astro_PLSCR1', 'Astro_WIF1',
       'EN_L2_3_IT_NTNG1', 'EN_L2_3_IT_PDGFD', 'EN_L3_5_IT_1_CUX2',
       'EN_L3_5_IT_1_PLSCR4', 'EN_L3_5_IT_2_DACH1', 'EN_L3_5_IT_2_HSPA1A',
       'EN_L3_5_IT_2_MET', 'EN_L3_5_IT_3', 'EN_L3_5_IT_3_HSPA1A', 'EN_L5_6_NP',
       'EN_L5_ET', 'EN_L6B', 'EN_L6_CT', 'EN_L6_CT_HSPA1A', 'EN_L6_IT_1',
       'EN_L6_IT_1_HSPA1A', 'EN_L6_IT_2', 'EN_L6_IT_2_HSPA1A', 'EN_NF',
       'Endo_IL1R1', 'Endo_ITIH5', 'Endo_PCSK5', 'IN_ADARB2_COL12A1',
       'IN_ADARB2_RAB37', 'IN_ADARB2_SV2C', 'IN_ADARB2_SYT10', 'IN_LAMP5_LHX6',
       'IN_LAMP5_LHX6_HSPA1A', 'IN_LAMP5_RELN', 'IN_LAMP5_RELN_HSPA1A',
       'IN_PVALB_ANOS1', 'IN_PVALB_CHC', 'IN_PVALB_CHRM2', 'IN_PVALB_HSPA1A',
       'IN_PVALB_STUM', 'IN_SST_EDNRA', 'IN_SST_EYA4', 'IN_SST_MAML3',
       'IN_SST_NPY', 'IN_SST_PRR16', 'IN_VIP_BCL11B', 'IN_VIP_SCML4',
       'IN_VIP_TRPC6', 'Immune_B', 'Immune_NK', 'Immune_PVM', 'Immune_Plasma',
       'Immune_T', 'Micro', 'OPC', 'OPC_CHRM3', 'OPC_GPR17', 'OPC_HAP1',
       'Oligo_OPALIN', 'Oligo_RBFOX1', 'PC_ADAMTS4', 'PC_STAC', 'PVM',
       'SMC_MYOCD', 'SMC_NRP1', 'VLMC_ABCA6', 'VLMC_DCDC2', 'VLMC_SLC4A4'
"""

from colour import Color
subclass_colors = dict(zip(subclass_palette.index,subclass_palette['color_hex'].values))
def generate_subtype_colors(subclass_colors, subtypes):
    """
    Generate a dictionary of colors for subtypes based on their subclass colors.
    
    :param subclass_colors: A dictionary mapping each subclass to its color in HEX format.
    :param subtypes: A list of subtypes, each prefixed with its subclass.
    :return: A dictionary mapping each subtype to a unique color.
    """
    subtype_colors = {}

    for subtype in subtypes:
        # Identify the subclass from the subtype
        subclass = subtype.split('_')[0]
        
        # Get the base color for the subclass
        base_color = Color(subclass_colors[subclass])
        
        # Create a slightly varied color for the subtype
        varied_color = base_color.luminance = base_color.luminance * 0.95  # Slightly darker
        subtype_colors[subtype] = varied_color.hex

    return subtype_colors
generate_subtype_colors(subclass_colors, df.columns.values)


#%%
d = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/PsychAD/adata_AD_sel_NPS_gxp.h5ad")
sc.pp.scale(d)
sc.pp.pca(d)
sc.tl.umap(d)
sc.pl.umap(d, color='subclass')

