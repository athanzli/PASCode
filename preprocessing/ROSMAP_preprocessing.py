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
DATA_PATH = '/home/che82/athan/ProjectPASCode/data/ROSMAP/'

#%%
###############################################################################
# load data
###############################################################################
data = scipy.io.mmread(DATA_PATH + 'raw/filtered_count_matrix.mtx')
data = scipy.sparse.csr_matrix(data)
obs = pd.read_csv(DATA_PATH + 'raw/filtered_column_metadata.txt',delimiter='\t')
gnames = pd.read_csv(DATA_PATH + 'raw/filtered_gene_row_names.txt',header=None)
gnames = gnames.values.flatten().tolist()
meta = pd.read_csv(DATA_PATH + 'raw/ROSMAP_metadata.csv', index_col=0)

# add donor info to obs
meta.index = meta['projid']
meta = meta.loc[obs['projid']]
obs.index = obs['TAG']
meta.index = obs.index
obs = pd.concat((obs, meta), axis=1)

#
adata = sc.AnnData(data.T, obs=obs)
adata.var['gene'] = list(gnames)
adata.var.index = list(gnames)
adata.obs.drop(columns=adata.obs.columns[adata.obs.columns.duplicated()], inplace=True)

adata.write_h5ad(DATA_PATH + 'raw/ROSMAP_raw.h5ad')

#%%
###############################################################################
# preprocess ROSMAP data and run DA tools
###############################################################################
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000) # NOTE
adata.raw = adata
adata = adata[:, adata.var.highly_variable] 

sc.pp.scale(adata)
PASCode.graph.build_graph(adata)

sc.pl.umap(adata, color=['broad.cell.type', 'diagnosis'])

# run DA&RRA to get aggregated labels
PASCode.da.agglabel(
    adata=adata,
    donor_col='individualID',
    cond_col='diagnosis',
    pos_cond='AD',
    neg_cond='CTL',
    methods=['milo','meld','daseq']
)
adata.write_h5ad(DATA_PATH + 'ROSMAP_AggLabel.h5ad')

#%% 
###############################################################################
# preprocess ROSMAP data with PsychAD HVGs (for GAT prediction for Fig. 2)
###############################################################################
adata = sc.read_h5ad(DATA_PATH + "ROSMAP_AggLabel.h5ad") # using the version already undergone log-transformation
adata = adata.raw
adata_ori = sc.read_h5ad(DATA_PATH + "raw/ROSMAP_raw.h5ad")

psychad_genes = pd.read_csv('/home/che82/athan/ProjectPASCode/data/PsychAD/PsychAD_hvg_3401.csv', index_col=0)
ovlp_genes = adata.var_names.intersection(psychad_genes.index)
rest_genes = psychad_genes.index.difference(ovlp_genes)
rest_genes_var = pd.DataFrame(index=rest_genes, columns=['gene'], data=rest_genes)

adata = anndata.AnnData(
    X = np.hstack([adata.X.toarray(), np.zeros((adata.shape[0], rest_genes.shape[0]))]),
    obs = adata_ori.obs,
    var = pd.concat([adata.var, rest_genes_var])
)
adata.var.pop('highly_variable')
adata = adata[:, psychad_genes.index]

sc.pp.scale(adata)

PASCode.graph.build_graph(adata)

sc.pl.umap(adata,color=['diagnosis', 'broad.cell.type'])

#%%
###############################################################################
# GAT prediction to get PAC scores for ROSMAP
###############################################################################
import torch
model = PASCode.model.GAT(in_channels=3401) # number of genes
model.load_state_dict(torch.load('/home/che82/athan/ProjectPASCode/train_model/trained_models/c02_model.pt'))
adata.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata))

#%%
###############################################################################
# add aggregated labels to ROSMAP data
###############################################################################
adata_label = sc.read_h5ad(DATA_PATH + 'ROSMAP_AggLabel.h5ad')
adata = sc.read_h5ad(DATA_PATH + 'ROSMAP_psychad_genes.h5ad')
assert all(adata.obs.index == adata_label.obs.index)
adata.obs['rra_pac'] = adata_label.obs['rra_pac']
adata.write_h5ad(DATA_PATH + 'ROSMAP_psychad_genes.h5ad')

#%%
###############################################################################
# use UMAP from original ROSMAP data for ROSMAP_psychad_genes.h5ad # NOTE
###############################################################################
adata = sc.read_h5ad(DATA_PATH + 'ROSMAP_psychad_genes.h5ad')
adata_ori = sc.read_h5ad(DATA_PATH + 'ROSMAP_AggLabel.h5ad')
adata.obsm['X_umap'] = adata_ori.obsm['X_umap']
adata.write_h5ad(DATA_PATH + 'ROSMAP_psychad_genes.h5ad')

