import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from PASCode import *

#%%
save_id = 'c125'
subid_col = 'SubID'
cond_col = 'c125x' # NOTE

print ('reading data...')
hpca = np.load('/home/che82/data/psychAD/MSSM_pca_regressed_harmony.npy')
meta = pd.read_csv('/home/che82/data/psychAD/MSSM_meta_obs-001.csv',index_col=0)
suball = pd.read_csv('/home/che82/data/psychAD/metadata.csv',index_col=0)

adata = anndata.AnnData(X=hpca, obs=meta)
sub = ['M95030']  # this one is not in the contrast list?
adata = adata[~adata.obs[subid_col].isin(sub),:]
cellinfo = suball.loc[adata.obs[subid_col], :]
cellinfo.index = adata.obs.index
allmeta = pd.concat([adata.obs, cellinfo],axis=1)
adata.obs = allmeta
mask = ~adata.obs[cond_col].isna()
adata = adata[mask]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

#%% ========================== preprocess =======================================
print('preprocessing...')
import warnings
from numba.core.errors import NumbaDeprecationWarning
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=TqdmWarning)

subinfo = subject_info(adata.obs, subid_col, columns=['r01x', cond_col])
print(subinfo[cond_col].value_counts())

#### build graph
adata.obsm['X_pca'] = adata.X
print("builidng graph...")
st = time.time()
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
print("building graph time cost (s): ", time.time() - st)

print("running umap...")
st = time.time()
sc.tl.umap(adata)
print("umap time cost (s): ", time.time() - st)

print("saving current anndata object...")
st = time.time()
chosen_col = [subid_col, cond_col, 'class', 'subclass', 'subtype']
adata.obs = adata.obs[chosen_col]
adata.write_h5ad('/home/che82/data/psychAD/contrasts/' + save_id + '.h5ad')
print("saving time cost (s): ", time.time() - st)


# %%
