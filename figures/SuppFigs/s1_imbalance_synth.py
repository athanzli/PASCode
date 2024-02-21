#%%
# %reload_ext autoreload 
# %autoreload 2

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

#%% ######################################### Calculate PACs #########################################
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data'
data_paths = sorted(glob.glob(DATA_PATH + '/synthetic/synth_imbalance/*h5ad'))

cond_col = 'syn_label'
pos_cond = 'cond1'
neg_cond = 'cond2'
subid_col = 'subid'

for i in range(len(data_paths)):
    print(f"\n ====================== Now processing the {i}th dataset ====================== \n")
    adata = sc.read_h5ad(data_paths[i])

    subinfo = PASCode.utils.subject_info(adata.obs, subid_col, columns=[cond_col])
    print(subinfo[cond_col].value_counts())

    adata.obs.rename(columns={'samples': 'subid', 'grd':'grd_truth'}, inplace=True)
    adata.obs['cond_bi'] = adata.obs[cond_col].map({pos_cond: 1, neg_cond: -1})
    adata.obs['true_pac'] = adata.obs['grd_truth'].map({'neg': -1, 'unk':0, 'pos':1})

    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata); 
    
    PASCode.da.run_milo(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_meld(adata, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_cna(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.da.run_daseq(adata, subid_col=subid_col, cond_col=cond_col, pos_cond=pos_cond, neg_cond=neg_cond)
    PASCode.rankaggr.rra(adata, score_cols=['milo','meld','cna', 'daseq'])

    PASCode.da.assign_pac_milo(adata, sfdr_thres=0.1)
    PASCode.da.assign_pac_meld(adata)
    PASCode.da.assign_pac_cna(adata, fdr_thres=0.1)
    adata.obs['rra_pac'] = PASCode.da.assign_pac(scores=adata.obs['rra_milo_meld_cna_daseq'].values, mode='cutoff', cutoff=0.5)

    print(adata.obs[['milo','milo_cell_lfc','meld','daseq','daseq_pac', 'cna', 'rra_milo_meld_cna_daseq', 'rra_pac', 'cond_bi']].corr())

    try:
        adata.uns.pop('NAM.T') # to avoid anndata saving error
    except:
        pass

    adata.write_h5ad(data_paths[i])

# %% ########################################### plot ###########################################
data_paths = sorted(glob.glob(DATA_PATH + '/synthetic/synth_imbalance/*h5ad'))
mlist = ['milo','meld','cna','daseq','rra'] # method list
metric_list = ['f1', 'p', 'r']
column_names = [f"{m}-{metric}" for metric in metric_list for m in mlist]
forplot =  pd.DataFrame(np.zeros((len(data_paths), len(mlist)*len(metric_list))), columns=column_names, index=[path.split('/')[-1].split('.')[-2] for path in data_paths])
#%% for other methods
for i in range(len(data_paths)):
    print(f"\n ====================== Now processing the {i}th dataset ====================== \n")
    adata = sc.read_h5ad(data_paths[i])
    subject_info = PASCode.utils.subject_info(adata.obs, subid_col, [cond_col])
    print(subject_info[cond_col].value_counts())

    for midx, m in enumerate(mlist):
        true_pac = adata.obs['true_pac'].astype(int).values
        report = sklearn.metrics.classification_report(
            true_pac, 
            adata.obs[f'{m}_pac'].astype(int).values, output_dict=True)
        forplot.iloc[i, midx + 0*len(mlist)] = report['macro avg']['f1-score']
        forplot.iloc[i, midx + 1*len(mlist)] = report['macro avg']['precision']
        forplot.iloc[i, midx + 2*len(mlist)] = report['macro avg']['recall']
        print(f"{m} f1-score: {report['macro avg']['f1-score']}, precision: {report['macro avg']['precision']}, recall: {report['macro avg']['recall']}")

forplot.to_csv('imbalance_forplot.csv')

# %% 
forplot = pd.read_csv('imbalance_forplot.csv', index_col=0)
res = forplot.copy()
row_to_move = res.iloc[:1]
res = res.drop(res.index[0])
part_before = res.iloc[:20] # NOTE 20
part_after = res.iloc[20:]  # NOTE 20
res = pd.concat([part_before, row_to_move, part_after])

methods = ['milo', 'meld', 'cna', 'daseq', 'rra']
Methods = ['Milo', 'MELD', 'CNA', 'DAseq', 'RRA']
metrics = ['f1', 'p', 'r']
Metrics = ['F1-score', 'Precision', 'Recall']

groups = ['2:24'] * 5 + ['6:24'] * 5 + ['12:24'] * 5 + ['18:24'] * 5 + ['24:24'] * 1 + \
    ['24:18'] * 5 + ['24:12'] * 5 + ['24:6'] * 5 + ['24:2'] * 5

fig, axes = plt.subplots(
    len(metrics), len(methods), 
    figsize=(30, 15),
    sharey='row'    
)

for i, (metric, Metric) in enumerate(zip(metrics, Metrics)):
    for j, (method, Method) in enumerate(zip(methods, Methods)):
        ax = axes[i, j]

        # sns.set(style="whitegrid")
        sns.boxplot(
            x=groups,
            y=res[f'{method}-{metric}'].values,
            medianprops=dict(color="orange", alpha=0.7),
            color='white', width=0.5, linewidth=1.5,
            ax=ax)

        sns.swarmplot(
            x=groups,
            y=res[f'{method}-{metric}'].values,
            dodge=True, size=3, legend=False, color='black',
            ax=ax)

        if i == 0:
            ax.set_title(f'{Method}')
            ax.title.set_fontsize(20)
        ## set fontsize for x and y ticks
        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.tick_params(axis='x', which='major', labelsize=15)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlabel('')
        ax.set_ylabel('')

        ## use grid
        ax.grid(True, axis='both', linestyle='--', linewidth=0.8)

plt.savefig('imbalance_benchmarking_res.pdf', dpi=600)

plt.tight_layout()
plt.show()

#%%
print(forplot[['milo-f1', 'meld-f1', 'cna-f1', 'daseq-f1', 'rra-f1']].var(axis=0))
