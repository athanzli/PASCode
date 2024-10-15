import numpy as np
import time
from sklearn.mixture import GaussianMixture

from .random_seed import *
set_seed(RAND_SEED)

from .utils import subject_info
from .rankaggr import rra

from .milopy.core import *
import meld
import cna, multianndata
import rpy2

import warnings

def agglabel(
    adata,
    subid_col,
    cond_col,
    pos_cond,
    neg_cond,
    da_methods=['milo','meld','daseq'],
):
    r"""
    CNA did not perform well in our benchmarking results. 
    We recommend using MILO, MELD and DAseq instead.
    """
    st = time.time()
    print("============================= DA and RRA... =============================")
    if 'milo' in da_methods:
        make_nhoods_prop = 0.05 if adata.shape[0] > 1e5 else 0.1 # according to Milo supp. note
        run_milo(adata, subid_col, cond_col, pos_cond, neg_cond, make_nhoods_prop=make_nhoods_prop)
    if 'meld' in da_methods:
        run_meld(adata, cond_col, pos_cond, neg_cond, beta=10, knn=15)
    if 'cna' in da_methods:
        run_cna(adata, subid_col, cond_col, pos_cond, neg_cond)
    if 'daseq' in da_methods:
        run_daseq(adata, subid_col, cond_col, pos_cond, neg_cond)

    rra(adata, da_methods=da_methods)
    print("============================= DA and RRA Time cost (s): ", np.round(time.time() - st, 2), " =============================\n")
    print(adata.obs['aggreg_label'].value_counts())
    return adata.obs['aggreg_label'].values

def _sort_clusters_by_values(clusters, values):
    """
    from https://github.com/KrishnaswamyLab/scprep/blob/master/scprep/utils.py

    Sort `clusters` in increasing order of `values`.

    Parameters
    ----------
    clusters : array-like
        An array of cluster assignments, like the output of
        a `fit_predict()` call.
    values : type
        An associated value for each index in `clusters` to use
        for sorting the clusters.

    Returns
    -------
    new_clusters : array-likes
        Reordered cluster assignments. `np.mean(values[new_clusters == 0])`
        will be less than `np.mean(values[new_clusters == 1])` which
        will be less than `np.mean(values[new_clusters == 2])`
        and so on.

    """
    if not len(clusters) == len(values):
        raise ValueError(
            "Expected clusters ({}) and values ({}) to be the "
            "same length.".format(len(clusters), len(values))
        )
    uniq_clusters = np.unique(clusters)
    means = np.array([np.mean(values[clusters == cl]) for cl in uniq_clusters])
    new_clust_map = {
        curr_cl: i for i, curr_cl in enumerate(uniq_clusters[np.argsort(means)])
    }
    return np.array([new_clust_map[cl] for cl in clusters])

def assign_pac(scores, mode='cutoff', cutoff=0.5, percentile=5):
    if mode == 'cutoff':
        scores = np.array(scores)
        assigned_pac = np.zeros_like(scores)
        assigned_pac[np.isnan(scores)] = np.nan
        assigned_pac[scores > cutoff] = 1
        assigned_pac[scores < -cutoff] = -1
    return assigned_pac

def assign_pac_milo(adata, sfdr_thres=0.1):
    # using avg. neighborhoods values for a cell by milo paper
    pos_mask = (adata.obs['milo_cell_lfc'] > 0) & (adata.obs['milo_cell_sfdr'] < sfdr_thres)
    neg_mask = (adata.obs['milo_cell_lfc'] < 0) & (adata.obs['milo_cell_sfdr'] < sfdr_thres)
    adata.obs['milo_pac'] = 0
    adata.obs.loc[pos_mask, 'milo_pac'] = 1
    adata.obs.loc[neg_mask, 'milo_pac'] = -1

def assign_pac_meld(adata):
    # cutoff (gaussian mixture, suggested by meld tutorial)
    mixture_model = GaussianMixture(n_components=3)
    classes = mixture_model.fit_predict(adata.obs['meld'].values.reshape(-1,1))
    classes = _sort_clusters_by_values(classes, adata.obs['meld'].values)
    adata.obs['meld_pac'] = 0
    adata.obs.loc[classes == 0, 'meld_pac'] = -1
    adata.obs.loc[classes == 1, 'meld_pac'] = 0
    adata.obs.loc[classes == 2, 'meld_pac'] = 1

def assign_pac_cna(adata, fdr_thres=0.1):
    # thres (https://github.com/immunogenomics/cna-display/blob/main/sepsis/sepsisresults.ipynb) and code for function 'association'
    if adata.uns['cna_fdrs']['fdr'].min() > fdr_thres:
        adata.obs['cna_pac'] = 0
    else:
        corr_thres = adata.uns['cna_fdrs'][adata.uns['cna_fdrs'].fdr <= fdr_thres].iloc[0].threshold
        adata.obs['cna_pac'] = 0
        adata.obs.loc[adata.obs['cna'] > corr_thres, 'cna_pac'] = 1
        adata.obs.loc[adata.obs['cna'] < - corr_thres, 'cna_pac'] = -1

def run_milo(adata, subid_col, cond_col, pos_cond, neg_cond, 
             design=None, model_contrast=None,
             make_nhoods_prop=0.1, use_rep='X_pca'):
    if 'connectivities' not in adata.obsp:
        # if anndata does not have X_pca then run PCA using scanpy
        if use_rep not in adata.obsm.keys():
            print("'use_pre' not found in adata.obsm.")
            print('Scaling...')
            sc.pp.scale(adata)
            print('Running PCA...')
            sc.pp.pca(adata, n_comps=50)
        print('Computing connectivities...')
        sc.pp.neighbors(adata)
    st = time.time()
    print("\n----------------------------- Milo started ... -----------------------------")
    print('Making neighborhoods...')
    make_nhoods(adata, prop=make_nhoods_prop)
    print('Counting neighborhoods...')
    count_nhoods(adata, sample_col=subid_col)
    print('Running differential abundance testing...')
    if design is not None:
        DA_nhoods(adata, design=design)
    else:
        DA_nhoods(adata, design='~'+cond_col)
    print("----------------------------- Milo Time cost (s): ", np.round(time.time() - st, 2) , " -----------------------------\n")
    # convert from neighborhood level to cell level
    warnings.filterwarnings("ignore")
    # print('Converting results from neighborhood level to cell level...')
    milo_res = adata.uns['nhood_adata'].obs
    cnbhd = adata.obsm['nhoods'].astype(int) # nbhd assignment of single cells # Use .toarray() only if adata.obsm["nhoods"] is sparse
    test_res_sum = cnbhd@milo_res.SpatialFDR.values
    count_nonzero = cnbhd.getnnz(axis=1)
    cell_sfdr = np.where(count_nonzero != 0, test_res_sum / count_nonzero, np.nan)
    test_res_sum = cnbhd@milo_res.logFC.values
    count_nonzero = cnbhd.getnnz(axis=1)
    cell_lfc = np.where(count_nonzero != 0, test_res_sum / count_nonzero, 0)

    # since milopy does not offer an interface to specify the reference level used within edgeR test setting, we need to correct the sign here 
    adata.obs['cond_bi'] = adata.obs[cond_col].map({pos_cond:1, neg_cond:0})
    mask = ~np.isnan(cell_lfc)
    sign = 2*(np.corrcoef(cell_lfc[mask], adata.obs['cond_bi'][mask])[0,1] > 0) - 1
    adata.obs['milo_cell_sfdr'] = cell_sfdr
    adata.obs['milo_cell_lfc'] = sign*cell_lfc
    adata.obs['milo'] = adata.obs['milo_cell_lfc']

def run_meld(adata, cond_col, pos_cond, neg_cond, beta=10, knn=15, use_rep='X_pca'):
    r"""
    According to the package documentation and source code, beta = 60 and knn = 5 are default parameter setting.
    However, we found that beta=10 and knn=15 lead to much more accurate and robust results for our datasets.
    We therefore recommend using beta=10 and knn=15 for most datasets.
    """
    # if anndata does not have X_pca then run PCA using scanpy
    if use_rep not in adata.obsm.keys():
        print("'use_pre' not found in adata.obsm.")
        print('Scaling...')
        sc.pp.scale(adata)
        print('Running PCA...')
        sc.pp.pca(adata, n_comps=50)

    st = time.time()    
    print("\n ----------------------------- MELD started ... -----------------------------")
    sample_densities = meld.MELD(beta=beta, knn=knn).fit_transform(adata.obsm[use_rep], adata.obs[cond_col].map({pos_cond: 0, neg_cond: 1})) # 0, 1 for ensuring sign
    print("----------------------------- MELD Time cost (s): ", np.round(time.time() - st, 2), " -----------------------------\n")

    adata.obsm['meld_res'] = sample_densities.values
    sample_likelihoods = meld.utils.normalize_densities(sample_densities)
    adata.obsm['meld_res_normalized'] = sample_likelihoods.values
    adata.obs['meld'] = sample_likelihoods.iloc[:,0].values * 2 - 1

def run_cna(adata, subid_col, cond_col, pos_cond, neg_cond, allow_low_sample_size=False, use_rep='X_pca'):
    if 'connectivities' not in adata.obsp:
        # if anndata does not have X_pca then run PCA using scanpy
        if use_rep not in adata.obsm.keys():
            print("'use_pre' not found in adata.obsm.")
            print('Scaling...')
            sc.pp.scale(adata)
            print('Running PCA...')
            sc.pp.pca(adata, n_comps=50)
        print('Computing connectivities...')
        sc.pp.neighbors(adata)

    st = time.time()
    print("\n----------------------------- CNA started ... -----------------------------")
    d = multianndata.MultiAnnData(adata, sampleid=subid_col)
    d.obs[cond_col] = d.obs[cond_col].map({pos_cond: 1, neg_cond: 0}).astype(int)
    d.obs_to_sample([cond_col])
    np.int = int # use this line to avoid numpy version issue
    cna_res = cna.tl.association(
        d,
        d.samplem[cond_col].astype('category').cat.codes,
        allow_low_sample_size=allow_low_sample_size)
    print("----------------------------- CNA Time cost (s): ", np.round(time.time() - st, 2) , " -----------------------------\n")
   
    adata.obs['cna'] = cna_res.ncorrs
    adata.uns['cna_fdrs'] = cna_res.fdrs

def run_daseq(adata, subid_col, cond_col, pos_cond, neg_cond, 
              k=[50,500,50], 
              use_rep='X_pca'):
    DAseq = rpy2.robjects.packages.importr('DAseq')
    rpy2.robjects.numpy2ri.activate()

    subinfo = subject_info(adata.obs, subid_col, [cond_col])

    # if anndata does not have X_pca then run PCA using scanpy
    if use_rep not in adata.obsm.keys():
        print("'use_pre' not found in adata.obsm.")
        print('Scaling...')
        sc.pp.scale(adata)
        print('Running PCA...')
        sc.pp.pca(adata, n_comps=50)
    X_pca = rpy2.robjects.numpy2ri.py2rpy(adata.obsm[use_rep])
    sampleid_cond1 = rpy2.robjects.StrVector(subinfo[subinfo[cond_col] == neg_cond].index.astype(str).tolist())
    sampleid_cond2 = rpy2.robjects.StrVector(subinfo[subinfo[cond_col] == pos_cond].index.astype(str).tolist())
    sampleid_each_cell = rpy2.robjects.StrVector(adata.obs[subid_col].astype(str).tolist())
    k = rpy2.robjects.IntVector(k)

    st = time.time()
    print("\n----------------------------- DAseq started ... -----------------------------")
    da_cells = DAseq.getDAcells(X=X_pca, 
                                cell_labels=sampleid_each_cell, 
                                labels_1=sampleid_cond1, 
                                labels_2=sampleid_cond2, 
                                k_vector=k,
                                do_plot=False)
    print("----------------------------- DA-seq Time cost (s): ", np.round(time.time() - st, 2), " -----------------------------\n")

    adata.obs['daseq'] = da_cells.rx2("da.pred")
    daseq_pac = np.zeros(adata.shape[0])
    daseq_pac[np.array(da_cells.rx2('da.up')) - 1] = 1
    daseq_pac[np.array(da_cells.rx2('da.down')) - 1] = -1
    adata.obs['daseq_pac'] = daseq_pac
