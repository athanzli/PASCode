from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from multianndata import MultiAnnData
import umap
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import milopy.core as milo
import milopy.plot as milopl
import milopy
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib_venn import venn2, venn3
import random
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import cna
import os

from .scACC.model import scACC

# relative_path = "./"
# absolute_path = os.path.abspath(relative_path)
# os.chdir(absolute_path)
# from scACC import *

np.random.seed(0)
random.seed(0)

def my_venn2(list1, list2, name1, name2, title):
    intersection_num = sum([1 for x, y in zip(list1, list2) if x==1 and x == y])
    venn2(subsets=( sum(list1) - intersection_num,
                    sum(list2) - intersection_num,
                    intersection_num),
                    set_labels=(name1, name2))
    plt.title(title)
    plt.show()

def my_venn3(list1, list2, list3, name1, name2, name3, title):
    region_7 = sum([1 for x, y, z in zip(list1, list2, list3) if x==1 and x == y and y == z])
    region_3 = sum([1 for x, y in zip(list1, list2) if x==1 and x == y]) - region_7
    region_5 = sum([1 for x, y in zip(list1, list3) if x==1 and x == y]) - region_7
    region_6 = sum([1 for x, y in zip(list2, list3) if x==1 and x == y]) - region_7
    region_4 = sum(list3) - region_7 - region_5 - region_6
    region_1 = sum(list1) - region_7 - region_3 - region_5
    region_2 = sum(list2) - region_7 - region_3 - region_6

    venn3(subsets=( region_1,
                    region_2,
                    region_3,
                    region_4,
                    region_5,
                    region_6,
                    region_7
                    ),
                    set_labels=(name1, name2, name3))
    plt.title(title)
    plt.show()

def summarize_1D_data(data): # 1d only
    quantile_25 = np.percentile(data, 25)
    quantile_75 = np.percentile(data, 75)
    median = np.median(data)
    variance = np.var(data)
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)
   
    plt.boxplot(data)

    print("25th percentile:", quantile_25)
    print("75th percentile:", quantile_75)
    print("Median:", median)
    print("Variance:", variance)
    print("Mean:", mean)
    print("Maximum:", maximum)
    print("Minimum:", minimum)

def plot_cell_score(x, y, scores, title, continuous=True): # for continuous 
    if continuous:
        colors = ["#0000ff", "#ffffff", "#ff0000"] # blue, white, red
        cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
        norm = plt.Normalize(scores.min(), scores.max())
        plt.figure(figsize=(10,10))
        ax = sns.scatterplot(x=x, y=y, hue=scores, 
                            palette=cmap, s=1, norm=norm, legend=False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

def plot_pac(x, y, pac, title=''):
    r"""
    Args:
        x ([type]): [description].
        y ([type]): [description].
        pac ([list, (1d)numpy.ndarray]): [...].
        method_name (str, optional): [description]. Defaults to ''.
        fdr_thres (float, optional): [description]. Defaults to .01.
    """
    # plt.figure(figsize=(10,10))
    # ax = sns.scatterplot(x=x, y=y, hue=is_pac, s=0.8, palette=['grey', 'red'])
    # ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()

    plt.figure(figsize=(6,6))
    ax = sns.scatterplot(x=x, y=y, hue=pac, s=3, palette=['blue', 'grey', 'red'])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def cluster_test(info, clustering_name, sampleid_name, 
                 phenotype_name, pos_phenotype_name): # TODO why does it result in different sc level plot results from the R version?
    r"""

    Returns:
        [pandas.DataFrame]: [cluster test results (e.g., logFC, FDR)]
    """
    pandas2ri.activate() # Enable automatic conversion from pandas to R
    base = importr('base') # Import R's "base" package
    edgeR = importr('edgeR')
    tab = pd.crosstab(info[sampleid_name], info[phenotype_name]) # NOTE by phenotype_name I mean the column name of phenotype in meta
    test_meta = pd.DataFrame({'Condition': (tab[pos_phenotype_name] > 0).astype(int)}) # e.g., pos_phenotype_name = 'AD
    r_test_meta = pandas2ri.py2rpy(test_meta) # Convert DataFrame to R's DataFrame
    leiden_model = robjects.r['model.matrix'](robjects.Formula('~Condition'), data=r_test_meta) # Create model matrix
    leiden_count = pandas2ri.py2rpy(pd.crosstab(info[clustering_name], info[sampleid_name])) # Create count matrix
    leiden_dge = edgeR.DGEList(counts=leiden_count, lib_size=robjects.r['log'](robjects.r['colSums'](leiden_count))) # Create DGE list
    leiden_dge = edgeR.estimateDisp(leiden_dge, leiden_model)
    leiden_fit = edgeR.glmQLFit(leiden_dge, leiden_model, robust=True)
    leiden_res = pandas2ri.rpy2py(robjects.r['as.data.frame'](edgeR.topTags(edgeR.glmQLFTest(leiden_fit, coef=2), sort_by='none', n=float('Inf'))))
    return leiden_res

def cluster_to_cell(cluster_test_res, clustering):
    r"""
    Args:
        cluster_test_res ([numpy.ndarrays]): [cluster test results 
            (e.g., logFC, or FDR), must be one-dimensional].
        clustering ([pandas.DataFrame]): [clustering results, constructed from
            pd.dummies() or the neighborhood matrix of the method itself].
    Returns:
        [numpy.ndarray]: [cell-level test results, one-dimensional].
    """
    test_res_sum = clustering@cluster_test_res
    count_nonzero = np.count_nonzero(clustering, axis=1)
    return np.array([test_res_sum[i] / count_nonzero[i] 
                     if count_nonzero[i]!=0 
                     else np.nan for i in range(len(test_res_sum))])

def run_scacc(X_train, 
              meta_train, 
              sampleid_name=None, 
              phenotype_name=None, # phenotype_name is the column name of phenotype in meta
              pos_phenotype_name=None,
              fdr_thres=0.01,
              latent_dim=3, 
              n_clusters=30, 
              lambda_cluster=1, 
              lambda_phenotype=1, 
              device='cpu',
              epoch_pretrain=15,
              epoch_train=15,
              batch_size=1024,
              lr_pretrain=1e-3,
              lr_train=1e-3,

              early_stopping=False,
              meta_test=None,

              alpha=1,
              dropout=.2,
              require_pretrain_phase=True,
              require_train_phase=True, 
              evaluation=False,
              plot_evaluation=False,
              id_train=None, X_test=None, y_test=None, id_test=None,
              fold_num=None,):
    r"""

    Args:

    Returns:
        scacc ([scACC]): [scACC model]
        pac_scacc ([numpy.ndarray]): [boolean 1d array, indicating PACs]
    """

    scacc = scACC(device=device, 
                  latent_dim=latent_dim, 
                  n_clusters=n_clusters, 
                  lambda_cluster=lambda_cluster, 
                  lambda_phenotype=lambda_phenotype,
                  dropout=dropout)

    # neg_phenotype_name = meta_train[phenotype_name].unique()[~(meta_train[phenotype_name].unique()==pos_phenotype_name)][0]
    y_train = meta_train[phenotype_name].factorize()[0] #.map({pos_phenotype_name: 1, neg_phenotype_name: 0}).values
    y_test = meta_test[phenotype_name].factorize()[0] #.map({pos_phenotype_name: 1, neg_phenotype_name: 0}).values
    scacc.train(X_train=X_train, 
                y_train=y_train,
                batch_size=batch_size, 
                epoch_pretrain=epoch_pretrain, 
                epoch_train=epoch_train,
                lr_pretrain=lr_pretrain, 
                lr_train=lr_train,
                early_stopping=False,
                X_test=X_test,
                y_test=y_test,)
    
    # clustering = scacc.get_cluster_assignments(X)
    # meta['scacc'] = clustering
    # cluster_test_res = cluster_test(info=meta, 
    #                                 clustering_name='scacc',
    #                                 sampleid_name=sampleid_name,
    #                                 phenotype_name=phenotype_name,
    #                                 pos_phenotype_name=pos_phenotype_name)
    # cell_fdr = cluster_to_cell(cluster_test_res['FDR'].values, pd.get_dummies(clustering))
    # cell_logfc = cluster_to_cell(cluster_test_res['logFC'].values, pd.get_dummies(clustering))
    # pac_scacc_pos = (cell_fdr < fdr_thres) & (cell_logfc < 0)
    # pac_scacc_neg = (cell_fdr < fdr_thres) & (cell_logfc > 0)
    return scacc

def run_cna(adata, return_pac=False, 
            sampleid_name=None, phenotype_name=None, 
            fdr_thres=0.01, n_neighbors=50, n_pcs=50):
    r"""
    Args:
        adata ([AnnData]): [AnnData object]
        phenotype_name ([str]): [Name of the column in meta that contains the phenotype]
        fdr_thres (float, optional): [FDR threshold]. Defaults to 0.01.
        n_neighbors (int, optional): [Number of neighbors]. Defaults to 50.
        n_pcs (int, optional): [Number of PCs]. Defaults to 50.
    Returns:
        [numpy.ndarray]: [PAC names]
    """
    adata = MultiAnnData(adata, sampleid=sampleid_name)
    adata.obs_to_sample([phenotype_name])
    if 'X_pca' not in adata.obsm.keys():
        sc.pp.pca(adata) # NOTE
    if 'connectivities' not in adata.obsp.keys():
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca') # NOTE
    res = cna.tl.association(adata, adata.samplem.AD)  # NOTE
    if not return_pac:
        return res
    # convert CNA results to single cell level 
    res=pd.DataFrame(np.zeros((len(adata), 2)), columns=['FDR', 'ncorr'])
    res.ncorr = res.ncorrs
    fdr_res = np.zeros(len(adata))
    res_fdrs_thresholds = res.fdrs.threshold.values
    res_fdrs_fdr = res.fdrs.fdr.values
    for i, score in enumerate(res.ncorrs):
        index = np.searchsorted(a=res_fdrs_thresholds, v=np.abs(score))
        index = min(index, len(res_fdrs_thresholds)-1) # ensure the index is within bounds
        fdr_res[i] = res_fdrs_fdr[index]
    res.FDR = fdr_res
    return res[res.FDR < fdr_thres].index.values

def run_milo(adata, return_pac_pos_neg=False, return_cell_scores=True, sampleid_name=None, phenotype_name=None, 
             fdr_thres=0.05, n_neighbors=15, n_pcs=50, make_nhoods_prop=0.1, visualize=False):
    r"""
    Args:
        adata ([anndata.AnnData]): [AnnData object]
        sampleid_name ([str]): [sample id column name]
        phenotype_name ([str]): [phenotype column name]
        n_neighbors ([int]): [number of neighbors in sc.pp.neighbors]
        n_pcs ([int]): [number of PCs in sc.pp.neighbors]
    Returns:
        [numpy.ndarray]: [boolean 1d array indicating PACs]
    """

    if 'X_pca' not in adata.obsm.keys():
        print('Running PCA...')
        sc.pp.pca(adata) # NOTE
    if 'connectivities' not in adata.obsp.keys():
        print('Building knn graph...')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca') # NOTE
    milo.make_nhoods(adata, prop=make_nhoods_prop) # NOTE 
    milo.count_nhoods(adata, sample_col=sampleid_name) # NOTE
    milo.DA_nhoods(adata, design='~'+phenotype_name) # NOTE

    if visualize:
        if 'X_umap' not in adata.obsm.keys():
            sc.tl.umap(adata)
        milopy.utils.build_nhood_graph(adata)
        milopy.plot.plot_nhood_graph(adata)
        sc.pl.umap(adata, color=[phenotype_name])
    # convert results to single cell level
    milo_res = adata.uns["nhood_adata"].obs
    clustering = adata.obsm['nhoods'].toarray().astype(int) # nbhd assignment of single cells # Use .toarray() only if adata.obsm["nhoods"] is sparse
    cell_fdr = cluster_to_cell(milo_res.SpatialFDR.values, clustering)
    cell_logfc = cluster_to_cell(milo_res.logFC.values, clustering)
    if return_pac_pos_neg is True:
        """
        cell_logfc < or >  0:
        In the formula notation used in the design matrix, the order of the levels in a factor often determines which group is used as the reference or baseline. In R, which this function seems to be written in, the default behavior is to use the first level as the reference when a factor is used in a formula.

        So, if the factor adata.obs[phenotype_name] has levels phenotype1 and phenotype2, and phenotype1 is the first level, then phenotype1 would be treated as the reference level. In that case:

        A positive logFC value would suggest that the feature is more abundant in phenotype2 compared to phenotype1 (the reference).
        A negative logFC value would suggest that the feature is less abundant in phenotype2 compared to phenotype1 (the reference).
        """
        milo_pac_pos = np.logical_and(~np.isnan(cell_fdr), cell_fdr < fdr_thres) & (cell_logfc < 0)
        milo_pac_neg = np.logical_and(~np.isnan(cell_fdr), cell_fdr < fdr_thres) & (cell_logfc > 0)
        return milo_pac_pos, milo_pac_neg, cell_fdr, cell_logfc
    milo_pac = np.logical_and(~np.isnan(cell_fdr), cell_fdr < fdr_thres)
    return milo_pac, 0, cell_fdr, cell_logfc

def run_leiden(X, meta, sampleid_name, phenotype_name, pos_phenotype_name, # TODO problems?? 
               pca, n_neighbors=50, n_pcs=50, resolution=0.9, visualize=False,
               fdr_thres=0.01):
    adata = anndata.AnnData(X=X, obs=meta)
    if pca is None:
        sc.pp.pca(adata) # NOTE
    else:
        adata.obsm['X_pca'] = pca
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=resolution)

    if visualize:
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['leiden'])

    # from clustering to cluster level testing results
    meta['leiden'] = adata.obs['leiden'].values.astype(int)
    test_res = cluster_test(info=meta, clustering_name='leiden', 
                            sampleid_name=sampleid_name, 
                            phenotype_name=phenotype_name, 
                            pos_phenotype_name=pos_phenotype_name)
    # from cluster level single cell level
    return cluster_to_cell(test_res.FDR.values, clustering=pd.get_dummies(meta['leiden']))


def evaluate_pac_prediction(mode='undirectional',
                            input_is_index=False, 
                            pac_true=None, 
                            pac_pred=None, 
                            pac_true_pos=None,
                            pac_true_neg=None,
                            pac_pred_pos=None,
                            pac_pred_neg=None,
                            cell_names=None):
    all_ind = np.arange(len(cell_names))
    if mode == 'undirectional':
        if input_is_index:
            y_pred = np.isin(all_ind, pac_pred)
            y_true = np.isin(all_ind, pac_true)
        else:
            y_pred = np.isin(cell_names, pac_pred)
            y_true = np.isin(cell_names, pac_true)
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        print('Recall: ', r)
        print('Precision: ', p)
        print('F-1 score: ', f)
        return r, p, f
    elif mode == 'directional':
        if input_is_index:
            y_pred_pos = np.isin(all_ind, pac_pred_pos)
            y_true_pos = np.isin(all_ind, pac_true_pos)
            y_pred_neg = np.isin(all_ind, pac_pred_neg)
            y_true_neg = np.isin(all_ind, pac_true_neg)
        else:
            y_pred_pos = np.isin(cell_names, pac_pred_pos)
            y_true_pos = np.isin(cell_names, pac_true_pos)
            y_pred_neg = np.isin(cell_names, pac_pred_neg)
            y_true_neg = np.isin(cell_names, pac_true_neg)
        r_pos = recall_score(y_true_pos, y_pred_pos)
        p_pos = precision_score(y_true_pos, y_pred_pos)
        f_pos = f1_score(y_true_pos, y_pred_pos)
        r_neg = recall_score(y_true_neg, y_pred_neg)
        p_neg = precision_score(y_true_neg, y_pred_neg)
        f_neg = f1_score(y_true_neg, y_pred_neg)
        print('Recall pos: ', r_pos)
        print('Precision pos: ', p_pos)
        print('F-1 score pos: ', f_pos)
        print('Recall neg: ', r_neg)
        print('Precision neg: ', p_neg)
        print('F-1 score neg: ', f_neg)
        return r_pos, p_pos, f_pos, r_neg, p_neg, f_neg