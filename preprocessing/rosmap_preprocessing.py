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
DATA_PATH = '/home/che82/data/rosmap/'

data = scipy.io.mmread(DATA_PATH + 'filtered_count_matrix.mtx')
data = scipy.sparse.csr_matrix(data)
meta = pd.read_csv(DATA_PATH + 'filtered_column_metadata.txt',delimiter='\t')
gns = pd.read_csv(DATA_PATH + 'filtered_gene_row_names.txt',header=None)
gns = gns.values.flatten().tolist()
dm = pd.read_csv(DATA_PATH + 'ROSMAP_metadata.csv', index_col=0) # add donor info to meta
dm.index = dm['projid']
mt = dm.loc[meta['projid']]
meta.index=meta['TAG']
mt.index=meta['TAG']
meta = pd.concat((meta,mt),axis=1)
adata = sc.AnnData(data.T, obs=meta)
adata.var['gene'] = list(gns)
adata.var.index = list(gns)
adata.obs.drop(columns=adata.obs.columns[adata.obs.columns.duplicated()], inplace=True)
adata.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_raw.h5ad')

# # Preprocessing
#%% preprocessing rosmap itself
adata = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_raw.h5ad")
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.normalize_total(adata, target_sum=1e4) # NOTE is using target_sum=1e6 which is logCPM, then the umap looks awful (why?)
sc.pp.log1p(adata) # NOTE
sc.pp.highly_variable_genes(adata)
adata.raw = adata
adata = adata[:, adata.var.highly_variable] 
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)


cond_col = 'diagnosis'
pos_cond = 'AD'
neg_cond = 'CTL'
subid_col = 'individualID'
PASCode.pac.run_milo(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
PASCode.pac.run_meld(adata, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond, beta=10, knn=15)
PASCode.pac.run_cna(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
PASCode.pac.run_daseq(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
PASCode.rankaggr.rra(adata, score_cols=['milo','meld','cna', 'daseq'])
PASCode.pac.assign_pac_milo(adata, sfdr_thres=0.1)
PASCode.pac.assign_pac_meld(adata)
PASCode.pac.assign_pac_cna(adata, fdr_thres=0.1)
adata.obs['rra_pac'] = PASCode.pac.assign_pac(scores=adata.obs['rra_milo_meld_cna_daseq'].values, mode='cutoff', cutoff=0.5)
assert all(adata.obs[['milo','milo_cell_lfc','meld','daseq','daseq_pac', 'cna', 'rra_milo_meld_cna_daseq', 'rra_pac', 'cond_bi']].corr() >= 0)

sc.pl.umap(adata,color=['diagnosis', 'broad.cell.type'])
adata.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap.h5ad')

#%% preprocessing rosmap with psychAD HVGs
adata = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_raw.h5ad")
adata.raw = adata
# read psychAD HVGs
psychad_genes = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/psychAD_hvg_3401.csv', index_col=0)
ovlp_genes = adata.var_names.intersection(psychad_genes.index)
rest_genes = psychad_genes.index.difference(ovlp_genes)
rest_genes_var = pd.DataFrame(index=rest_genes, columns=['gene'], data=rest_genes)
adata = anndata.AnnData(
    X = np.hstack([adata.X.toarray(), np.zeros((adata.shape[0], rest_genes.shape[0]))]),
    obs = adata.obs,
    var = pd.concat([adata.var, rest_genes_var])
)
adata = adata[:, psychad_genes.index]

sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pl.umap(adata,color=['diagnosis', 'broad.cell.type'])
adata.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')

#%% for fig2 RF classifier
adata = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')
adata_ref = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/c02_100v100_gb.h5ad')
sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.ingest(adata, adata_ref, obs='subtype')
adata.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')

# %%
