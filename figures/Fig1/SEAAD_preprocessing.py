#%%
###############################################################################
# setup
###############################################################################
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PASCode
DATA_PATH = '/home/che82/athan/ProjectPASCode/data/SEA-AD/'

#%%
###############################################################################
# load data and subsample donors
###############################################################################
adata = sc.read_h5ad(DATA_PATH + "SEA-AD/SEAAD_DLPFC_RNAseq_final-nuclei.2023-07-19.h5ad")

cond_col = 'Cognitive Status'
pos_cond = 'Dementia'
neg_cond = 'No dementia'
donor_col = 'Donor ID'
bk_col = 'Braak'
class_col = 'Class'
subclass_col = 'Subclass'

dinfo = PASCode.utils.subject_info(
    obs = adata.obs,
    donor_col=donor_col,
    columns=[bk_col, 'Sex', cond_col],
)
# remove reference
dinfo = dinfo[~((dinfo[bk_col] == 'Reference') | (dinfo[cond_col] == 'Reference'))]
# subsample donors
dinfo = dinfo[~dinfo.index.isin(dinfo[(dinfo['Sex']=='Male') & (dinfo[cond_col]=='No dementia')][-2:].index)]
dinfo[cond_col].value_counts()
mask = adata.obs[donor_col].isin(dinfo.index).values
adata = adata[mask]
adata.obs = adata.obs[[donor_col, cond_col, bk_col, 'Sex', subclass_col, class_col]]

#%%
###############################################################################
# load data and subsample donors
###############################################################################
sc.pp.neighbors(adata, use_rep='X_scVI')

# these two lines are needed to avoid design bug
adata.obs['Cognitive_Status'] = adata.obs[cond_col]
adata.obs['Cognitive_Status'] = adata.obs['Cognitive_Status'].map({pos_cond:'Dementia', neg_cond:'No_dementia'}).values

cond_col = 'Cognitive_Status'
neg_cond = 'No_dementia'

PASCode.da.run_milo(
    adata,
    donor_col, 'Cognitive_Status', 'Dementia', 'No_dementia',
    make_nhoods_prop=0.05) 
PASCode.da.run_meld(
    adata,
    cond_col, pos_cond, neg_cond,
    beta=10, knn=15,
    use_rep='X_scVI')
PASCode.da.run_daseq(
    adata, donor_col, cond_col, pos_cond, neg_cond,
    k=[50,500,50],
    use_rep='X_scVI')
PASCode.da.rra(adata, score_cols=['milo', 'meld', 'daseq'])
adata.obs['rra_pac'] = PASCode.da.assign_pac(adata.obs['rra_milo_meld_daseq'].values)

#%% 
###############################################################################
# gene overlapping with psychAD
###############################################################################
psychad_genes = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/psychAD_hvg_3401.csv', index_col=0)
ovlp_genes = adata.var_names.intersection(psychad_genes.index)

adata = adata[:, ovlp_genes]

rest_genes = psychad_genes.index.difference(ovlp_genes)
rest_genes_var = pd.DataFrame(index=rest_genes, columns=['gene'], data=rest_genes)

adata = anndata.AnnData(
    X = scipy.sparse.hstack([adata.X, scipy.sparse.csr_matrix((adata.shape[0], rest_genes.shape[0]), dtype=np.float32)]),
    obs = adata.obs,
    var = pd.concat([adata.var, rest_genes_var]),
)
adata = adata[:, psychad_genes.index]

adata.raw = adata
sc.pp.scale(adata)

#%%
###############################################################################
# GAT prediction to get PAC scores for SEA-AD
###############################################################################
import torch
model = PASCode.model.GAT(in_channels=3401) # number of genes
model.load_state_dict(torch.load('/home/che82/athan/ProjectPASCode/train_model/trained_models/c02_model.pt'))
adata.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata))

adata.write_h5ad(DATA_PATH + 'SEAAD.h5ad')
