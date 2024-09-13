# %%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.append('..')
import PASCode

import scanpy as sc
import pandas as pd
import numpy as np
import torch

PASCode.random_seed.set_seed(0)

# %%
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/'

subinfo_all = pd.read_csv(DATA_PATH + 'metadata.csv', index_col=0)

cond_col = 'c02x'
pos_cond = 'AD' 
neg_cond = 'Control'
subid_col = 'SubID'

# %%
adata = sc.read_h5ad(DATA_PATH + 'c02.h5ad')
adata_pac = sc.read_h5ad(DATA_PATH + 'c02_100v100_gb.h5ad')
adata_pac.obs['rra_pac'] = PASCode.pac.assign_pac(
    scores = adata_pac.obs['rra_milo_meld_daseq'].values, # NOTE
    mode='cutoff', cutoff=0.5); print(adata_pac.obs['rra_pac'].value_counts())
adata.obs['rra_pac'] = np.nan
adata.obs.loc[adata_pac.obs.index, 'rra_pac'] = adata_pac.obs['rra_pac'].values
print(adata.obs['rra_pac'].value_counts())
adata.obs['c02_100v100_mask'] = adata.obs.index.isin(adata_pac.obs.index)
subinfo = PASCode.utils.subject_info(adata.obs, 
                                     subid_col=subid_col, 
                                     columns=[cond_col, 'Sex'])
print(subinfo.groupby([cond_col, 'Sex']).size().unstack())

#%% ########################################### prep data for GNN ###########################################
# 10% subjects for val
subinfo = PASCode.utils.subject_info(adata_pac.obs, 
                                     subid_col=subid_col, 
                                     columns=[cond_col, 'Sex'])
mp = subinfo[(subinfo['Sex'] == 'Male') & (subinfo[cond_col] == pos_cond)] \
        .sample(n=5).index.values
fp = subinfo[(subinfo['Sex'] == 'Female') & (subinfo[cond_col] == pos_cond)] \
        .sample(n=5).index.values
mn = subinfo[(subinfo['Sex'] == 'Male') & (subinfo[cond_col] == neg_cond)] \
        .sample(n=5).index.values
fn = subinfo[(subinfo['Sex'] == 'Female') & (subinfo[cond_col] == neg_cond)] \
        .sample(n=5).index.values
sub_sel = np.concatenate([mp, fp, mn, fn])
adata_pac.obs['val_mask'] = adata_pac.obs[subid_col].isin(sub_sel).values
adata_pac.obs['train_mask'] = ~adata_pac.obs['val_mask']
adata.obs['train_mask'] = adata.obs.index \
    .isin(adata_pac.obs.index[adata_pac.obs['train_mask']])
adata.obs['val_mask'] = adata.obs.index \
    .isin(adata_pac.obs.index[adata_pac.obs['val_mask']])

# NOTE
adata.write_h5ad(DATA_PATH + 'c02.h5ad')
adata_pac.write_h5ad(DATA_PATH + 'c02_100v100_gb.h5ad')

subinfo = PASCode.utils.subject_info(adata.obs[adata.obs['train_mask']], 
                                     subid_col=subid_col, 
                                     columns=[cond_col, 'Sex'])
print(subinfo.groupby([cond_col,'Sex']).size().unstack())
subinfo = PASCode.utils.subject_info(adata.obs[adata.obs['val_mask']], 
                                     subid_col=subid_col, 
                                     columns=[cond_col, 'Sex'])
print(subinfo.groupby([cond_col,'Sex']).size().unstack())

# %%
data = PASCode.model.Data().adata2gdata(
    adata,
    y=adata.obs['rra_pac'].values + 1,
    trn_mask=adata.obs['train_mask'].values,
    val_mask=adata.obs['val_mask'].values)
data_loader = PASCode.model.Data().gdata2batch(
    data,
    batch_size=128,
    num_parts=2048,
    shuffle=True)

# %%
model = PASCode.model.GAT(
    in_channels=adata.X.shape[1], out_channels=64, num_class=3, heads=4)

best_model = PASCode.model.Trainer(model=model, device='cuda').train(
    trn_data_loader=data_loader, data_val=data, val_data_loader=None,
    max_epoch=100, lr=1e-3, lr_decay=[2, 0.5], early_stopping=10, weight_decay=1e-3, # NOTE
    class_weight=[1, 1, 1])
model = best_model

torch.save(model.state_dict(), './trained_models/c02_model.pt')

#%%
model.load_state_dict(torch.load('./trained_models/c02_model.pt'))

#%% predict
adata_tst = sc.read_h5ad(DATA_PATH + 'c02.h5ad')
adata_tst.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata_tst))
adata_tst.write_h5ad(DATA_PATH + 'c02.h5ad')

adata_tst = sc.read_h5ad(DATA_PATH + 'c03.h5ad')
adata_tst.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata_tst))
adata_tst.write_h5ad(DATA_PATH + 'c03.h5ad')

adata_tst = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')
adata_tst.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata_tst))
adata_tst.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/rosmap/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')

adata_tst = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/SeaAD/seaad.h5ad')
adata_tst.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata_tst))
adata_tst.write_h5ad('/home/che82/athan/PASCode/code/github_repo/data/SeaAD/seaad.h5ad')

#%% ###########################################  latent space ###########################################
import umap
from sklearn.decomposition import PCA

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'
tst_cids = [
    'PsychAD/c02', 
    'SeaAD/seaad', 
    'rosmap/rosmap_ovlpgenes_with_psychAD_contrasts']
for tst_cid in tst_cids:
    adata = sc.read_h5ad(DATA_PATH + tst_cid + '.h5ad')

    data = PASCode.model.Data().adata2gdata(adata)

    with torch.no_grad():
        x1 = torch.nn.functional.elu(model.conv1(data.x, data.edge_index))
        x2 = torch.nn.functional.elu(model.conv2(x1, data.edge_index))
    adata.obsm['model_layer2'] = x2.numpy()

    latent_pca = PCA(n_components=50).fit_transform(adata.obsm['model_layer2'])
    adata.obsm['model_layer2_umap'] = umap.UMAP(n_neighbors=50).fit_transform(latent_pca)

    adata.write_h5ad(DATA_PATH + tst_cid + '.h5ad')


