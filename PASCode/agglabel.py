#%%
###############################################################################
# Script version of agglabel
###############################################################################

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PASCode

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--donor_col', type=str, required=True, help='subject ID column name in anndata.obs')
parser.add_argument('--cond_col', type=str, required=True, help='Condition column name')
parser.add_argument('--pos_cond', type=str, required=True, help='Positive condition name')
parser.add_argument('--neg_cond', type=str, required=True, help='Negative condition name')
parser.add_argument('--file_path', type=str, required=True, help='Path to read anndata')
args = parser.parse_args()

cond_col = args.cond_col
donor_col =  args.donor_col
pos_cond = args.pos_cond
neg_cond = args.neg_cond
file_path = args.file_path
save_path = args.file_path

"""
e.g.
cond_col = 'c90x'
donor_col = 'SubID'
pos_cond = 'Sleep_WeightGain_Guilt_Suicide'
neg_cond = 'Control'
file_path = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/c90_63v63.h5ad'
"""

# load  data
import scanpy as sc
print("Reading data....")
adata = sc.read_h5ad(file_path)

# run DA methods and RRA
PASCode.pac.run_milo(adata, donor_col, cond_col, pos_cond, neg_cond)
PASCode.pac.run_meld(adata, cond_col, pos_cond, neg_cond, beta=10, knn=15)
PASCode.pac.run_daseq(adata, donor_col, cond_col, pos_cond, neg_cond)

PASCode.rankaggr.rra(adata, score_cols=['milo','meld','daseq'])

adata.obs['rra_pac'] = PASCode.pac.assign_pac(
    adata.obs['rra_milo_meld_daseq'].values,
    mode='cutoff',
    cutoff=0.5)
print(adata.obs['rra_pac'].value_counts())

# save resutls
print("Writing data...")
adata.write_h5ad(save_path)
print("Done.")
