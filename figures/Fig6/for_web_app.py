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
contrasts = ['c02x', 'c90x', 'c91x', 'c92x']
pos_conds = ['AD', 'Sleep_WeightGain_Guilt_Suicide', 'WeightLoss_PMA', 'Depression_Mood']
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
 'EN_NF', 'IN_ADARB2', 'IN_LAMP5_LHX6', 'IN_LAMP5_RELN', 'IN_PVALB',
 'IN_PVALB_CHC', 'IN_SST', 'IN_VIP', 'Endo', 'PC', 'SMC', 'VLMC']
celltypes = adata.obs.groupby(['subclass', 'class']).size().unstack().loc[subclasses, classes]
# print(celltypes)
celltypes.to_csv("class_subclass_cellnumber.csv")
celltypes.to_csv("celltypes_order.csv")
class_subclass_map = dict(zip(celltypes.columns, [celltypes.index[np.where(celltypes[col] > 0)[0]] for col in celltypes.columns]))

subclass_palette = pd.read_csv("../subclass_palette.csv", index_col=0)
subclass_int_map = dict(zip(subclass_palette.index, 1 + np.arange(len(subclass_palette))))

#%%
###############################################################################
# webapp df
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


contrasts = ['c02x', 'r01x', 'c90x', 'c91x', 'c92x']
adatas = []
for i in range(len(contrasts)):
    adatas.append(sc.read_h5ad(DATA_PATH + f"PsychAD/{contrasts[i][:-1]}_only_obs_obsm.h5ad"))

for i in range(len(contrasts)):
    adatas[i].obs[f'{contrasts[i][:-1]}_pac_score'] = adatas[i].obs['pac_score'].values
    adatas[i].obs[f'{contrasts[i][:-1]}_PAC'] = PASCode.pac.assign_pac(
        adatas[i].obs[f'{contrasts[i][:-1]}_pac_score'].values, mode='cutoff', cutoff=0.5)
    adatas[i].obs[f'{contrasts[i][:-1]}_umap0'] = adatas[i].obsm['X_umap'][:,0]
    adatas[i].obs[f'{contrasts[i][:-1]}_umap1'] = adatas[i].obsm['X_umap'][:,1]
    adatas[i].obs[f'{contrasts[i][:-1]}_model_layer2_umap0'] = adatas[i].obsm['model_layer2_umap'][:,0]
    adatas[i].obs[f'{contrasts[i][:-1]}_model_layer2_umap1'] = adatas[i].obsm['model_layer2_umap'][:,1]

common_cols = ['Sex', 'SubID', 'class', 'r01x', 
               'subclass', 'subtype']
spe_cols = []
for i in range(len(contrasts)):
    spe_cols.append(
        [f'{contrasts[i]}',
         f'{contrasts[i][:-1]}_umap0',
         f'{contrasts[i][:-1]}_umap1',
         f'{contrasts[i][:-1]}_model_layer2_umap0', 
         f'{contrasts[i][:-1]}_model_layer2_umap1', 
         f'{contrasts[i][:-1]}_pac_score',
         f'{contrasts[i][:-1]}_PAC']
    )

df = pd.concat([
    adatas[0].obs[common_cols],
    adatas[1].obs[common_cols],
    adatas[2].obs[common_cols],
    adatas[3].obs[common_cols],
    adatas[4].obs[common_cols]
], axis=0)
df['barcode'] = df.index
df = df.drop_duplicates()
df.drop(columns=['barcode'], inplace=True)
df

def remove_dupicated_cols(df):
    df_T = df.T
    df_T = df_T.loc[~df_T.index.duplicated(keep='first')]
    df_unique = df_T.T
    return df_unique

for i, col in enumerate(spe_cols):
    df[col] = np.nan
    adatas[i].obs = remove_dupicated_cols(adatas[i].obs)
    df.loc[adatas[i].obs.index, col] = adatas[i].obs.loc[:, col]

df['Ethnicity'] = allmeta.loc[df.index, 'Ethnicity']
df['Age'] = allmeta.loc[df.index, 'Age']
df['Ethnicity'] = allmeta.loc[df.index, 'Ethnicity']
df['Ethnicity'] = allmeta.loc[df.index, 'Ethnicity']


df.to_csv("webapp_df.csv")


# %%
