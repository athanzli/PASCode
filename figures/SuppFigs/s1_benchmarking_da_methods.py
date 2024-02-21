#%%
%reload_ext autoreload 
%autoreload 2

import glob
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/che82/athan/PASCode/code/github_repo/')
import PASCode
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data'

#%%
###############################################################################
# Calculate PACs
###############################################################################
data_paths = sorted(glob.glob(DATA_PATH + '/synthetic/s1.1/*h5ad'))

cond_col = 'syn_label'
pos_cond = 'cond1'
neg_cond = 'cond2'
subid_col = 'subid'

for i in range(len(data_paths)):
    print(f"\n ====================== Now processing the {i}th dataset ====================== \n")

    adata = sc.read_h5ad(data_paths[i])

    adata.obs.rename(columns={'samples': 'subid', 'grd':'grd_truth'}, inplace=True)
    adata.obs['cond_bi'] = adata.obs[cond_col].map({pos_cond: 1, neg_cond: -1})
    adata.obs['true_pac'] = adata.obs['grd_truth'].map({'neg': -1, 'unk':0, 'pos':1})
    
    subinfo = PASCode.utils.subject_info(adata.obs, subid_col, columns=[cond_col])
    print(subinfo[cond_col].value_counts())

    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    PASCode.da.run_milo(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_meld(adata, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_cna(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_daseq(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.rankaggr.rra(adata, score_cols=['milo','meld','cna', 'daseq'])

    PASCode.da.assign_pac_milo(adata, sfdr_thres=0.1)
    PASCode.da.assign_pac_meld(adata)
    PASCode.da.assign_pac_cna(adata, fdr_thres=0.1)
    adata.obs['rra_pac'] = PASCode.da.assign_pac(
        scores=adata.obs['rra_milo_meld_cna_daseq'].values, mode='cutoff', cutoff=0.5)

    corr = adata.obs[['milo','milo_cell_lfc','meld','daseq','daseq_pac', 'cna', 'rra_milo_meld_cna_daseq', 'rra_pac', 'cond_bi']].corr()
    assert all(corr > 0), "corr matrix has negative values"

    try:
        adata.uns.pop('NAM.T') # to avoid anndata saving error
    except:
        pass
    adata.write_h5ad(data_paths[i])

#%% 
###############################################################################
# store pac metrics results in a dataframe for plotting
###############################################################################
mlist = ['milo','meld','cna','daseq','rra'] # method list
metric_list = ['f1-score', 'precision', 'recall']
column_names = [f"{m}_{metric}" for metric in metric_list for m in mlist]
forplot =  pd.DataFrame(np.zeros((len(data_paths), len(mlist)*len(metric_list))), 
                        columns=column_names, index=[path.split('/')[-1] for path in data_paths])

for i in range(len(data_paths)):
    print(f"\n ====================== Now processing the {i}th dataset ====================== \n")
    adata = sc.read_h5ad(data_paths[i])
    for j in range(len(metric_list)):
        for m in range(len(mlist)):
            res = sklearn.metrics.classification_report(
                adata.obs['true_pac'],
                adata.obs[f'{mlist[m]}_pac'].values,
                output_dict=True)
            forplot.iloc[i, j*5 + m] = res['macro avg'][metric_list[j]]

forplot.to_csv('/home/che82/athan/PASCode/code/github_repo/figures/s1/da_rra_benchmarking_res.csv')

#%%
###############################################################################
# plot
###############################################################################
forplot = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/figures/s1/da_rra_benchmarking_res.csv', index_col=0)

methods = ['milo', 'meld', 'cna', 'daseq', 'rra']
Methods = ['Milo', 'MELD', 'CNA', 'DAseq', 'RRA']
metrics = ['f1-score', 'precision', 'recall']
n_methods = len(methods)
n_metrics = len(metrics)
n_conds = 5 # see the synthetic data
n_lfcs = 3 # 0.7, 0.8, 0.9

import matplotlib.patches as mpatches
fig, axes = plt.subplots(nrows=n_metrics, ncols=4, figsize=(20, 15)) # NOTE
# colors = sns.color_palette()
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
# colors = ['#ffffb3', '#cee1e1', '#c8baf6', '#bebada', '#fb8072', 'tab:red']
patches = [mpatches.Patch(color=colors[i], label=label) for i, label in enumerate(['Milo', 'MELD' ,'CNA', 'DAseq', 'RRA'])]

for i in range(4): # 12:12, 9:12, 6:12, 3:12
    for metric_idx in range(n_metrics):
        df = pd.DataFrame(columns=['Group', 'Score', 'Method'], index=range((n_lfcs*n_conds)*n_methods)) # 15 scores, n_methods tools
        df['Method'] = np.array([np.repeat(m, n_lfcs*n_conds) for m in Methods]).flatten()
        df['Group'] = (['0.7'] * n_conds + ['0.8'] * n_conds + ['0.9'] * n_conds) * n_methods
        df['Score'] = forplot.iloc[
            np.linspace(0,60,16)[:-1]+i, 
            [j+n_methods*metric_idx for j in range(n_methods)]].T.values.flatten()
        g = sns.boxplot(x='Group', y='Score', hue='Method', data=df, ax=axes[metric_idx, i], width=0.6, palette=colors)
        g.set(ylim=(0,1));  g.legend_.remove(); g.set_xlabel(''); g.set_ylabel('')
        g.grid(True)
        ## add vertical lines to each x tick
        for tick in g.get_xticks():
            g.axvline(tick, linestyle='--', alpha=0.3, color='grey')
        # if metric_idx != 2:
        # g.set_xticks([])
        g.set_yticks(np.arange(0, 1.2, 0.2))

fig.text(0.12, 1.01, '12:12', va='center', ha='left', fontsize=25)
fig.text(0.36, 1.01, '9:12', va='center', ha='left', fontsize=25)
fig.text(0.60, 1.01, '6:12', va='center', ha='left', fontsize=25)
fig.text(0.84, 1.01, '3:12', va='center', ha='left', fontsize=25)

fig.text(1.00, 0.83, 'F1-score', va='center', ha='left', fontsize=25, rotation=-90)
fig.text(1.00, 0.52, 'Precision', va='center', ha='left', fontsize=25, rotation=-90)
fig.text(1.00, 0.18, 'Recall', va='center', ha='left', fontsize=25, rotation=-90)

plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19

lgd=fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(patches), fontsize=20)

plt.tight_layout()
plt.savefig("./da_benchmarking_res.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)


# %%
# #=== TEMP DEL 
# adata.obs.drop(columns=adata.obs.columns[adata.obs.columns.duplicated()], inplace=True)
# adata.obs.drop(columns=['individualID', 'diagnosis', 'TAG',  'tsne1', 'tsne2', 'pre.cluster',
#                         'name', 'V1', 'fastq', 'Study', 'msex', 'educ',
#                         'race', 'spanish', 'apoe_genotype', 'age_at_visit_max',
#                         'age_first_ad_dx', 'age_death', 'cts_mmse30_first_ad_dx',
#                         'cts_mmse30_lv', 'pmi', 'braaksc', 'ceradsc', 'leiden2', 'cogdx', 'dcfdx_lv',],inplace=True)
# #===