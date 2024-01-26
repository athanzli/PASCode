import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PASCode

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--subid_col', type=str, required=True, help='subject ID column name in anndata.obs')
parser.add_argument('--cond_col', type=str, required=True, help='Condition column name')
parser.add_argument('--pos_cond', type=str, required=True, help='Positive condition name')
parser.add_argument('--neg_cond', type=str, required=True, help='Negative condition name')
parser.add_argument('--read_path', type=str, required=True, help='Path to read anndata')
parser.add_argument('--save_path', type=str, required=True, help='Path to save anndata')
args = parser.parse_args()

cond_col = args.cond_col
subid_col =  args.subid_col
pos_cond = args.pos_cond
neg_cond = args.neg_cond
read_path = args.read_path
save_path = args.save_path

"""
cid = 'c90_63v63'
cond_col = 'c90x'
subid_col = 'SubID'
pos_cond = 'Sleep_WeightGain_Guilt_Suicide'
neg_cond = 'Control'
read_path = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/c90_63v63.h5ad'
"""

# prep data
import scanpy as sc
print("reading data....")
adata = sc.read_h5ad(read_path)

# run DA methods and RRA 
# NOTE if you just masked adata which originally had a graph, then you also 
#   just masked some nodes in the graph, which may cause an error in milo's make nhoods.
#   To avoid this error, construct the graph again after masking. This is also 
#   beneficial for CNA, which also depends on the graph.
PASCode.pac.run_milo(adata, subid_col, cond_col, pos_cond, neg_cond)
PASCode.pac.run_meld(adata, cond_col, pos_cond, neg_cond, beta=10, knn=15)
# PASCode.pac.run_cna(adata, subid_col, cond_col, pos_cond, neg_cond)
PASCode.pac.run_daseq(adata, subid_col, cond_col, pos_cond, neg_cond)

PASCode.rankaggr.rra(adata, score_cols=['milo','meld','daseq'])

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(adata.obs['rra_milo_meld_daseq'].values, bins=100); plt.show()
adata.obs['rra_pac'] = PASCode.pac.assign_pac(adata.obs['rra_milo_meld_daseq'].values,
                mode='cutoff', cutoff=0.5); print(adata.obs['rra_pac'].value_counts())

## test corr
print(adata.obs[['milo','meld','cna','daseq','daseq_pac','cond_bi', 'rra_pac']].corr())

# sns.scatterplot(x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], hue=adata.obs['class'], s=1); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left'); plt.show()
# sns.scatterplot(x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], hue=adata.obs[cond_col], palette={pos_cond: 'red', neg_cond: 'blue'}, s=1); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left') ; plt.show()

# sns.histplot(adata.obs['milo'].values, bins=100);     plt.show()
# sns.histplot(adata.obs['meld'].values, bins=100);     plt.show()
# sns.histplot(adata.obs['cna'].values, bins=100);      plt.show()
# sns.histplot(adata.obs['daseq'].values, bins=100);    plt.show()
# sns.histplot(adata.obs['combined'].values, bins=100); plt.show()

# plot_cell_scores(title='Milo', x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], scores=adata.obs['milo'], s=3, continuous=True,            );plt.show()
# plot_cell_scores(title='MELD', x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], scores=adata.obs['meld'], s=3, continuous=True,            );plt.show()
# plot_cell_scores(title='CNA', x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], scores=adata.obs['cna'], s=3, continuous=True,              );plt.show()
# plot_cell_scores(title='DAseq', x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], scores=adata.obs['daseq'], s=3, continuous=True,          );plt.show()
# plot_cell_scores(title='Combined', x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], scores=adata.obs['combined'], s=3, continuous=True,    );plt.show()
# sns.scatterplot(x=adata.obsm['X_umap'][:,0], y=adata.obsm['X_umap'][:,1], hue=adata.obs['combined_pac'], size=adata.obs['combined_pac'], sizes={-1: 1, 0: 1, 1: 1}, palette={-1: 'blue', 0: 'grey', 1: 'red'});plt.show()

# save resutls
print("writing data...")
adata.write_h5ad(save_path)
print("done.")