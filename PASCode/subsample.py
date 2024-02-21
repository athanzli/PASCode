#%%
###############################################################################
# Function version of subsample donors
###############################################################################
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import PASCode
import numpy as np

import warnings
from numba.core.errors import NumbaDeprecationWarning
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=TqdmWarning)

def subsample_donors(
    adata,
    subsample_num,
    donor_col,
    cond_col,
    pos_cond,
    neg_cond,
    sex_col=None,
    mode='random', # 'random' or 'top'
):
    print("Before donor subsampling:")
    dinfo = PASCode.utils.subject_info(
        obs=adata.obs,
        subid_col=donor_col,
        columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
    print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())

    print("Donor subsampling: ", subsample_num)
    pos_donor_num = int(subsample_num.split(':')[0])
    neg_donor_num = int(subsample_num.split(':')[1])
    if sex_col is None:
        if mode == 'random':
            sel_pos = dinfo[dinfo[cond_col]==pos_cond].sample(n=pos_donor_num).index
            sel_neg = dinfo[dinfo[cond_col]==neg_cond].sample(n=neg_donor_num).index
        elif mode == 'top':
            sel_pos = dinfo[dinfo[cond_col]==pos_cond].sort_values(by='cell_num', ascending=False)[:pos_donor_num].index
            sel_neg = dinfo[dinfo[cond_col]==neg_cond].sort_values(by='cell_num', ascending=False)[:neg_donor_num].index
        chosen_donors = np.concatenate((sel_pos, sel_neg))
    else:
        pos_hf_num = pos_donor_num // 2
        neg_hf_num = neg_donor_num // 2
        df = dinfo.groupby([cond_col, sex_col]).size().unstack()
        female = df.columns[0] # NOTE we ignored a strict correspondence here
        male = df.columns[1]
        if df.loc[pos_cond].min() >= pos_hf_num:
            posm_num = pos_hf_num
            posfm_num = pos_hf_num
        else:
            posm_num = df.loc[pos_cond, male]
            posfm_num = pos_donor_num - posm_num
        if df.loc[neg_cond].min() >= neg_hf_num:
            negm_num = neg_hf_num
            negfm_num = neg_hf_num
        else:
            negm_num = df.loc[neg_cond, male]
            negfm_num = neg_donor_num - negm_num
        
        if mode == 'random':
            pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sample(n=posm_num).index
            pos_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sample(n=posfm_num).index
            neg_m_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==male)].sample(n=negm_num).index
            neg_fm_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==female)].sample(n=negfm_num).index
        elif mode == 'top':
            pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sample(n=posm_num).index
            pos_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sample(n=posfm_num).index
            neg_m_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==male)].sample(n=negm_num).index
            neg_fm_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==female)].sample(n=negfm_num).index

        chosen_donors = np.concatenate((pos_m_donors, pos_fm_donors, neg_m_donors, neg_fm_donors))
    # adata_ori = adata.copy() # adata=adata_ori
    adata = adata[adata.obs[donor_col].isin(chosen_donors)]

    print("After donor subsampling:")
    dinfo = PASCode.utils.subject_info(
        obs=adata.obs,
        subid_col=donor_col,
        columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
    print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())

    return adata

# #%%
# ###############################################################################
# # Script version of subsample donors
# ###############################################################################
# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# import PASCode
# import scanpy as sc
# import numpy as np

# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--donor_col', type=str, required=True, help='Donor ID column name in anndata.obs')
# parser.add_argument('--cond_col', type=str, required=True, help='Condition column name in anndata.obs')
# parser.add_argument('--pos_cond', type=str, required=True, help='Positive condition name in condition column')
# parser.add_argument('--neg_cond', type=str, required=True, help='Negative condition name in condition column')
# parser.add_argument('--file_path', type=str, required=True, help='Path to read anndata')
# parser.add_argument('--sex_col', type=str, required=False, help='Sex column name in anndata.obs')
# parser.add_argument('--male', type=str, required=False, help='Male sex name in sex column')
# parser.add_argument('--female', type=str, required=False, help='Female sex name in sex column')
# args = parser.parse_args()

# cond_col = args.cond_col
# donor_col =  args.donor_col
# pos_cond = args.pos_cond
# neg_cond = args.neg_cond
# file_path = args.file_path
# save_path = args.file_path
# sex_col = args.sex_col
# male = args.male
# female = args.female

# import warnings
# from numba.core.errors import NumbaDeprecationWarning
# from tqdm import TqdmWarning
# warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
# warnings.filterwarnings("ignore", category=TqdmWarning)

# adata = sc.read_h5ad(file_path)

# print("Before donor subsampling:")
# dinfo = PASCode.utils.subject_info(
#     obs=adata.obs,
#     subid_col=donor_col,
#     columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
# print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())

# if args.donor_sample is not None:
#     print("Donor subsampling: ", args.donor_sample)
#     pos_donor_num = int(args.donor_sample.split(':')[0])
#     neg_donor_num = int(args.donor_sample.split(':')[1])
#     if sex_col is None:
#         sel_pos = dinfo[dinfo[cond_col]==pos_cond].sample(n=pos_donor_num).index
#         # sel_pos = dinfo[dinfo[cond_col]==pos_cond].sort_values(by='cell_num', ascending=False)[:pos_donor_num].index
#         mask_pos = adata.obs[donor_col].isin(sel_pos)
#         sel_neg = dinfo[dinfo[cond_col]==pos_cond].sample(n=pos_donor_num).index
#         # sel_neg = dinfo[dinfo[cond_col]==neg_cond].sort_values(by='cell_num', ascending=False)[:neg_donor_num].index
#         chosen_donors = np.concatenate((sel_pos, sel_neg))
#     else:
#         pos_donor_num_half = int(pos_donor_num // 2)
#         neg_donor_num_half = int(neg_donor_num // 2)
#         # pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sort_values(by='cell_num', ascending=False)[:pos_donor_num_half].index
#         pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sample(n=pos_donor_num_half).index
#         pos_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sample(n=pos_donor_num_half).index
#         neg_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sample(n=neg_donor_num_half).index
#         neg_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sample(n=neg_donor_num_half).index
#         chosen_donors = np.concatenate((pos_m_donors, pos_fm_donors, neg_m_donors, neg_fm_donors))
#     adata_ori = adata.copy()
#     adata = adata[adata.obs[donor_col].isin(chosen_donors)]

# print("After donor subsampling:")
# dinfo = PASCode.utils.subject_info(
#     obs=adata.obs,
#     subid_col=donor_col,
#     columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
# print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())
