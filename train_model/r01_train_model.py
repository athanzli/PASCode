# %%
import sys
sys.path.append('..')
import PASCode

import scanpy as sc
import pandas as pd
import numpy as np
import torch

PASCode.random_seed.set_seed(0)

# %%
DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

subinfo_all = pd.read_csv(DATA_PATH + 'metadata.csv', index_col=0)
adata = sc.read_h5ad(DATA_PATH + 'PsychAD/r01.h5ad')
adata_pac = sc.read_h5ad(DATA_PATH + 'PsychAD/r01_30v30_gb.h5ad')

cond_col = 'r01x_bi'
pos_cond = 'bk6' 
neg_cond = 'bk0'
subid_col = 'SubID'

# %%
adata.obs['rra_pac'] = np.nan
adata_pac.obs['rra_pac'] = PASCode.pac.assign_pac(adata_pac.obs['rra_milo_meld_daseq'].values)
adata.obs.loc[adata_pac.obs.index, 'rra_pac'] = adata_pac.obs['rra_pac'].values
print(adata.obs['rra_pac'].value_counts())
adata.obs['r01_30v30_mask'] = adata.obs.index.isin(adata_pac.obs.index)
subinfo = PASCode.utils.subject_info(adata.obs, 
                                     subid_col='SubID', 
                                     columns=['r01x', 'Sex'])
print(subinfo.groupby(['r01x','Sex']).size().unstack())

#%% ########################################### prep data for GNN ###########################################
# 10% subjects for val
subinfo = PASCode.utils.subject_info(adata_pac.obs, 
                                     subid_col='SubID', 
                                     columns=[cond_col, 'Sex', 'r01x'])
mpos = subinfo[(subinfo['Sex'] == 'Male') & (subinfo[cond_col] == pos_cond)] \
        .sample(n=2).index.values
fmpos = subinfo[(subinfo['Sex'] == 'Female') & (subinfo[cond_col] == pos_cond)] \
        .sample(n=1).index.values
mneg = subinfo[(subinfo['Sex'] == 'Male') & (subinfo[cond_col] == neg_cond)] \
        .sample(n=1).index.values
fmneg = subinfo[(subinfo['Sex'] == 'Female') & (subinfo[cond_col] == neg_cond)] \
        .sample(n=2).index.values
sub_sel = np.concatenate([mpos, fmpos, mneg, fmneg])
adata_pac.obs['val_mask'] = adata_pac.obs['SubID'].isin(sub_sel).values
adata_pac.obs['train_mask'] = ~adata_pac.obs['val_mask']
adata.obs['train_mask'] = adata.obs.index \
    .isin(adata_pac.obs.index[adata_pac.obs['train_mask']])
adata.obs['val_mask'] = adata.obs.index \
    .isin(adata_pac.obs.index[adata_pac.obs['val_mask']])

subinfo = PASCode.utils.subject_info(adata.obs[adata.obs['train_mask']], 
                                     subid_col='SubID', 
                                     columns=['r01x', 'Sex'])
print(subinfo.groupby(['r01x','Sex']).size().unstack())
subinfo = PASCode.utils.subject_info(adata.obs[adata.obs['val_mask']], 
                                     subid_col='SubID', 
                                     columns=['r01x', 'Sex'])
print(subinfo.groupby(['r01x','Sex']).size().unstack())

# %%
data = PASCode.model.Data().adata2gdata(adata, 
                                 y=adata.obs['rra_pac'].values + 1, 
                                 trn_mask=adata.obs['train_mask'].values, 
                                 val_mask=adata.obs['val_mask'].values)
data_loader = PASCode.model.Data().gdata2batch(data, 
                                        batch_size=128, 
                                        num_parts=2048, 
                                        shuffle=True)

# %%
model = PASCode.model.GAT(in_channels=adata.X.shape[1], 
                          out_channels=64, num_class=3, heads=4)

best_model = PASCode.model.Trainer(model=model, device='cuda').train(
    trn_data_loader=data_loader, data_val=data, val_data_loader=None,
    max_epoch=100, lr=1e-3, lr_decay=[2, 0.5], early_stopping=10, weight_decay=1e-3)

#%%
model = best_model

model.load_state_dict(torch.load('./trained_models/r01_model.pt'))

with torch.no_grad():
    adata.obs['pac_score'] = model.predict(data)

#%%
with torch.no_grad():
    x1 = torch.nn.functional.elu(model.conv1(data.x, data.edge_index))
    x2 = torch.nn.functional.elu(model.conv2(x1, data.edge_index))
adata.obsm['model_layer2'] = x2.numpy()
import umap
from sklearn.decomposition import PCA
latent_pca = PCA(n_components=50).fit_transform(adata.obsm['model_layer2'])
adata.obsm['model_layer2_umap'] = umap.UMAP(n_neighbors=50) \
    .fit_transform(latent_pca)

# %%
torch.save(model.state_dict(), './trained_models/r01_model.pt') # NOTE
adata.write_h5ad(DATA_PATH + 'PsychAD/r01.h5ad')
adata_pac.write_h5ad(DATA_PATH + 'PsychAD/r01_30v30_gb.h5ad')
