import scanpy as sc
import anndata

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/'
adata = sc.read_h5ad(DATA_PATH + "c02_100v100_gb.h5ad")

#%% [markdown]
## add gene expression data
adata_gxp = sc.read_h5ad('/media/che82/hechenfon/pead_freeze25/datasets2.5_M_selgenes2.h5ad')
ind = [adata_gxp.obs.index.get_loc(i) for i in adata.obs.index]
adata.obsm['X_pca_regressed_harmony'] = adata.X
adata = anndata.AnnData(X=adata_gxp.X[ind, :], 
                        obs=adata.obs, 
                        var=adata_gxp.var, 
                        uns=adata.uns, 
                        obsm=adata.obsm, 
                        obsp=adata.obsp)
adata.raw = adata
sc.pp.scale(adata)

adata.write_h5ad(DATA_PATH + "c02_100v100_gb.h5ad")
