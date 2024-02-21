#%%
# %reload_ext autoreload
# %autoreload 2
import scanpy as sc
import anndata
import sys
import pandas as pd
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PASCode

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cond_col', type=str, help='condition column name in PsychAD, e.g. c02x', required=True)
parser.add_argument('--donor_col', type=str, help='donor column name in PsychAD, e.g. SubID', required=False, default='SubID') # for PsychAD
parser.add_argument('--pos_cond', type=str, help='positive condition name in PsychAD, e.g. AD in c02x', required=False)
parser.add_argument('--neg_cond', type=str, help='negative condition name in PsychAD, e.g. Control in c02x', required=False)
parser.add_argument('--save_path', type=str, help='path to save the anndata object, e.g. ./c02_100v100_gb.h5ad', required=True)
parser.add_argument('--run_da', type=bool, help='whether to run DA and RRA', default=False, required=True)
parser.add_argument('--subsample_num', type=str, help='number of donors to sample from each condition, e.g. 100:100 for c02x', required=False)
parser.add_argument('--sex_col', type=str, help='Sex column in PsychAD', required=False, default='Sex')
args = parser.parse_args()

donor_col = args.donor_col
cond_col = args.cond_col
pos_cond = args.pos_cond
neg_cond = args.neg_cond
subsample_num = args.subsample_num
sex_col = args.sex_col
save_path = args.save_path
run_da = args.run_da

#%%
###############################################################################
# Build anndata
###############################################################################
# choose the corresponding contrast
DATA_PATH = '/home/che82/athan/ProjectPASCode/data/PsychAD/'
print ('Loading data...')
hpca = np.load(DATA_PATH + 'MSSM_pca_regressed_harmony.npy')
obs = pd.read_csv(DATA_PATH + 'MSSM_meta_obs-001.csv',index_col=0) # read original cell meta data
assert hpca.shape[0] == obs.shape[0]
adata = anndata.AnnData(X=hpca, obs=obs)
rmds = ['M95030']  # this one is not in the contrast list
adata = adata[~adata.obs[donor_col].isin(rmds),:]

meta = pd.read_csv(DATA_PATH + 'metadata_0828.csv', index_col=0) # 1494 donors, not including M95030
dinfo = meta.loc[adata.obs[donor_col].values, :]
dinfo.index = adata.obs.index
meta_obs = pd.concat([adata.obs, dinfo],axis=1)
adata.obs = meta_obs
mask = ~pd.isna(adata.obs[cond_col])
adata = adata[mask]

# add gene expression
adata.obsm['X_pca_regressed_harmony'] = adata.X
print('Loading gene expression...')
adata_gxp = sc.read_h5ad('/media/che82/hechenfon/pead_freeze25/datasets2.5_M_selgenes2.h5ad') # log-transformed
ind = [adata_gxp.obs.index.get_loc(i) for i in adata.obs.index]
adata = anndata.AnnData(X=adata_gxp.X[ind, :],
                        obs=adata.obs,
                        var=adata_gxp.var,
                        uns=adata.uns,
                        obsm=adata.obsm,
                        obsp=adata.obsp)
adata.raw = adata
adata.obs = adata.obs[[
    'SubID', cond_col, 'class', 'subclass', 'subtype', 'Sex', 'c28x', 'r01x'
]]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

#%%
###############################################################################
# subsample
###############################################################################
if subsample_num is not None:
    if cond_col == 'r01x':
        adata = adata[(adata.obs['r01x']==0) | (adata.obs['r01x']==6)]
    adata = PASCode.subsample.subsample_donors(
        adata=adata,
        subsample_num=subsample_num,
        donor_col=donor_col,
        cond_col=cond_col,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        sex_col=sex_col,
        mode='random',
    )

sc.pp.scale(adata)

#%%
###############################################################################
# build graph
###############################################################################
adata.obsm['X_pca'] = adata.obsm['X_pca_regressed_harmony']
adata = PASCode.graph.build_graph(adata=adata)

#%%
###############################################################################
# run DA to get aggregated labels
###############################################################################
if run_da:
    PASCode.da.agglabel(
        adata,
        donor_col,
        cond_col,
        pos_cond,
        neg_cond,
        methods=['milo','meld','daseq']
    )

#%%
###############################################################################
# save
###############################################################################
adata.write_h5ad(save_path)
