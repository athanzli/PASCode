#%%
%reload_ext autoreload 
%autoreload 2

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata
import sys
sys.path.append('/home/che82/athan/PASCode/code/github_repo/')
import PASCode

#%% load data
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

#%% read
adata = sc.read_h5ad(DATA_PATH + "SEA-AD/SEAAD_DLPFC_RNAseq_final-nuclei.2023-07-19.h5ad")
#%% donor preprocessing
cond_col = 'Cognitive Status'
pos_cond = 'Dementia'
neg_cond = 'No dementia'
subid_col = 'Donor ID'
bk_col = 'Braak'
class_col = 'Class'
subclass_col = 'Subclass'

subinfo = PASCode.utils.subject_info(
    obs = adata.obs,
    subid_col=subid_col,
    columns=[bk_col, 'Sex', cond_col],
)
# remove reference
subinfo = subinfo[~((subinfo[bk_col] == 'Reference') | (subinfo[cond_col] == 'Reference'))]
# subsample donors
subinfo = subinfo[~subinfo.index.isin(subinfo[(subinfo['Sex']=='Male') & (subinfo[cond_col]=='No dementia')][-2:].index)]
subinfo[cond_col].value_counts()

#%%
mask = adata.obs[subid_col].isin(subinfo.index).values
adata = adata[mask]
adata.obs = adata.obs[[subid_col, cond_col, bk_col, 'Sex', subclass_col, class_col]]
#%% run 4 tools
import sys
sys.path.append('/home/che82/athan/PASCode/code/github_repo/')

import PASCode
# NOTE if you just masked adata which originally had a graph, then you also 
#   just masked some nodes in the graph, which may cause an error in milo's make nhoods.
#   To avoid this error, construct the graph again after masking.
sc.pp.neighbors(adata, use_rep='X_scVI')

# these two lines are needed to avoid design bug
adata.obs['Cognitive_Status'] = adata.obs[cond_col]
adata.obs['Cognitive_Status'] = adata.obs['Cognitive_Status'].map({pos_cond:'Dementia', neg_cond:'No_dementia'}).values
PASCode.pac.run_milo(adata, subid_col, 'Cognitive_Status', 'Dementia', 'No_dementia', 
                    make_nhoods_prop=0.05) 

PASCode.pac.run_meld(adata, cond_col, pos_cond, neg_cond, beta=10, knn=15,
                    use_rep='X_scVI')
PASCode.pac.run_cna(adata, subid_col, cond_col, pos_cond, neg_cond)
PASCode.pac.run_daseq(adata, subid_col, cond_col, pos_cond, neg_cond, k=[50,500,50],
                     use_rep='X_scVI')
PASCode.pac.rra(adata, score_cols=['milo', 'meld', 'daseq'])
PASCode.pac.rra(adata, score_cols=['milo', 'meld', 'cna', 'daseq'])
adata.obs['rra_pac'] = PASCode.pac.assign_pac(adata.obs['rra_milo_meld_daseq'].values)

#%% gene overlapping with psychAD
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

#%% for fig2 RF classifier
adata_ref = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/c02_100v100_gb.h5ad')
sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.ingest(adata, adata_ref, obs='subclass')
adata.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/SEA-AD/seaad.h5ad')

sc.pl.umap(adata, color=['Subclass', 'subclass'])

#%% save
adata.write_h5ad("/home/che82/athan/PASCode/code/github_repo/data/SEA-AD/seaad.h5ad")





#%% ################################################## backup

adata = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/SEA-AD/seaad_no_gxp.h5ad")
adata2 = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/SEA-AD/seaad_no_gxp2.h5ad")

adata.obs
import scanpy as sc

adata2 = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/ROSMAP/rosmap.h5ad")

adata3=  sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/ROSMAP/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad")

adata2

adata3

adata


adata2 = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/ROSMAP/rosmap_raw.h5ad")

np.intersect1d(adata2.var.index, psychad_genes.index).shape


d = sc.read_h5ad("/home/che82/Downloads/SEAAD_DLPFC_RNAseq_final-nuclei.2023-07-19.h5ad")


np.intersect1d(d.var.index, psychad_genes.index).shape



