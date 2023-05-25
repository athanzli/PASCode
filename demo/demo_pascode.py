#%%
%reload_ext autoreload 
%autoreload 2

import scanpy as sc
import pandas as pd
import pickle as pkl
import numpy as np

import os
pascode_dir = '/home/che82/athan/pascode/github'
os.chdir(pascode_dir)
from PASCode import PASCode
from PASCode.utils import *

#%% read data
with open('/home/che82/project/mit-ana/half_traintest.pkl', 'rb') as f:
    data = pkl.load(f)
meta_train = data[0]
meta_test = data[2]
X_train = data[1]
X_test = data[3]
meta = pd.concat([data[0], data[2]])
X = pd.concat([data[1], data[3]])
meta['AD'] = meta['diagnosis'].apply(lambda x: 1 if x == 'AD' else 0)
meta_train['AD'] = meta_train['diagnosis'].apply(lambda x: 1 if x == 'AD' else 0)
meta_test['AD'] = meta_test['diagnosis'].apply(lambda x: 1 if x == 'AD' else 0)
meta.tsne1 = meta.tsne1.values.astype(float)
meta.tsne2 = meta.tsne2.values.astype(float)
meta_train.tsne1 = meta_train.tsne1.values.astype(float)
meta_train.tsne2 = meta_train.tsne2.values.astype(float)
meta_test.tsne1 = meta_test.tsne1.values.astype(float)
meta_test.tsne2 = meta_test.tsne2.values.astype(float)
meta_test_imb = meta_test[np.logical_or(meta_test['diagnosis'] == 'CTL', meta_test['projid'] == 20170043)] # NOTE
X_test_imb = X_test[np.logical_or(meta_test['diagnosis'] == 'CTL', meta_test['projid'] == 20170043)] # NOTE

#%% prepare
sampleid_name = 'projid' # NOTE change here per dataset. Column name for sample id in adata.obs.
phenotype_name = 'diagnosis' # NOTE change here per dataset. Column name for phenotype in adata.obs.
pos_phenotype_name = 'AD' # NOTE change here per dataset. Positive phenotype name in adata.obs[phenotype_name].
fdr_thres = .01
adata = sc.AnnData(X=X, obs=meta)
adata_train = sc.AnnData(X=X_train, obs=meta_train)
adata_test = sc.AnnData(X=X_test, obs=meta_test)
#%% run pascode
##
pac_true = run_milo(adata_test, return_pac=True, sampleid_name=sampleid_name, phenotype_name=phenotype_name)
pc = PASCode(sampleid_name=sampleid_name, 
             phenotype_name=phenotype_name, 
             pos_phenotype_name=pos_phenotype_name)
pc.fit(adata_train, fdr_thres=fdr_thres)
pac_pred = pc.predict(adata_test.X, cell_names=adata_test.obs.index.values)
##
evaluate_pac_predicton(pac_true, pac_pred, cell_names=meta_test.index.values)
##
plot_pac(x=meta_test.tsne1.values, y=meta_test.tsne2.values, 
         is_pac=np.isin(meta_test.index.values, pac_true), 
         title='Milo PACs on test set')
plot_pac(x=meta_test.tsne1.values, y=meta_test.tsne2.values, 
         is_pac=np.isin(meta_test.index.values, pac_pred), 
         title=' Predicted PACs on test set')
##
z = pc.get_latent_space(adata.X)