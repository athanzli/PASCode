import scanpy as sc
DATA_PATH = "/home/che82/athan/ProjectPASCode/data/"

adata2 = sc.read_h5ad(DATA_PATH + 'SEA-AD/SEAAD.h5ad')
adata3 = sc.read_h5ad(DATA_PATH + 'ROSMAP/ROSMAP_psychad_genes.h5ad')

#%%
###############################################################################
# first write out the correspondence table in python
###############################################################################
tab = adata2.obs.groupby(['Subclass','subclass']).size().unstack()
ntab2 = tab.div(tab.sum(axis=1), axis=0)
ntab2.to_csv('perc.s8.SEAAD.csv')
tab = adata3.obs.groupby(['Subcluster','subclass']).size().unstack()
ntab3 = tab.div(tab.sum(axis=1), axis=0)
ntab3.to_csv('perc.s8.ROSMAP.csv')

#%%
###############################################################################
# then plot in R (see R script)
###############################################################################
pass 


