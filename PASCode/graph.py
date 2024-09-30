#%%
###############################################################################
# Function version of building graph
###############################################################################
import time
import scanpy as sc

def build_graph(
    adata,
    use_rep='X_pca',
    run_umap=True,
):
    r"""
    Assuming adata has been preprocessed (including standard scaling) with adata.X as the expression matrix.
    If the input data is not a subsampled dataset, we recommend running PCA first.
    Otherwise, you can directly use the precomputed PCA results.

    Args:
        adata (AnnData): Annotated data matrix.
        use_rep (str, optional): The representation to use for building graph. Default: 'X_pca'.
        run_neighbors_for_subsampled_data (bool, optional): Whether to run neighbors for subsampled data. Default: True.
        run_umap (bool, optional): Whether to run umap. Default: True.
    """
    if use_rep == 'X_pca':
        if 'X_pca' not in adata.obsm.keys():
            print("Scaling data...")
            sc.pp.scale(adata)
            print("Running PCA...")
            sc.pp.pca(adata)
        else:
            print("Using anndata.obsm['X_pca'] as rep...")
    else:
        assert use_rep in adata.obsm.keys(), \
            f"Error: use_rep {use_rep} not in adata.obsm.keys()."
        print(f"Using {use_rep} as rep...")

    if ('neighbors' not in adata.uns.keys()) or ('connectivities' not in adata.obsp.keys()):
        print("Builidng graph...")
        st = time.time()
        sc.pp.neighbors(adata, use_rep=use_rep) # default is 50 PCs
        print(f"Building graph time cost (s): {(time.time() - st):2f}.")

    if run_umap:
        print("Running umap...")
        st = time.time()
        sc.tl.umap(adata)
        print(f"Umap time cost (s): {(time.time() - st):2f}")

# #%%
# ###############################################################################
# # Script version of building graph
# ###############################################################################

# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# import scanpy as sc
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--file_path', type=str, help='Path to load the .h5ad file.')
# parser.add_argument('--n_pcs', type=int, help='Number of PCs to use for building graph, e.g. 50', default=50)
# args = parser.parse_args()

# anndata_name = args.anndata_name
# save_name = anndata_name
# n_pcs = args.n_pcs

# import time
# import warnings
# from numba.core.errors import NumbaDeprecationWarning
# from tqdm import TqdmWarning
# warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
# warnings.filterwarnings("ignore", category=TqdmWarning)

# adata = sc.read_h5ad(args.file_path)

# if 'X_pca' not in adata.obsm.keys():
#     print("Running PCA...")
#     sc.tl.pca(adata)
# else:
#     print("Using anndata.obsm['X_pca'] as rep...")

# print("Builidng graph...")
# st = time.time()
# sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep='X_pca') # default is 50 PCs
# print("Building graph time cost (s): ", time.time() - st)

# print("Running umap...")
# st = time.time()
# sc.tl.umap(adata)
# print("Umap time cost (s): ", time.time() - st)

# print("Saving anndata...")
# st = time.time()
# adata.write_h5ad(args.file_path)
# print("Saving time cost (s): ", time.time() - st)
