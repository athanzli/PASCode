#%%
import numpy as np
import sys
sys.path.append('/home/che82/athan/PASCode/code/github_repo/')
import PASCode
import torch
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

PASCode.random_seed.set_seed(42)

#%%
###############################################################################
# Load data
###############################################################################
adata = sc.read_h5ad("/home/che82/athan/PASCode/code/github_repo/data/ROSMAP/rosmap.h5ad")

cond_col = 'diagnosis'
pos_cond = 'AD'
neg_cond = 'CTL'
subid_col = 'individualID'
sex_col = 'msex'

# adata.obs['rra_pac'].value_counts()
# PASCode.da.assign_pac(
#     scores=adata.obs['rra_milo_meld_cna_daseq'].values, mode='cutoff', cutoff=0.5)
# adata.obs['rra_pac'].value_counts()

dinfo = PASCode.utils.subject_info(
    obs = adata.obs,
    subid_col=subid_col,
    columns=[cond_col, 'msex']
)
dinfo[cond_col].value_counts()

#%%
###############################################################################
# splitting
###############################################################################
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=6, test_size=8.0/48, random_state=42)
dinfo['cond_sex_col'] = dinfo[cond_col] + dinfo[sex_col].astype(str)
val_donors = []
for i, (trn_idx, val_idx) in enumerate(sss.split(np.zeros(len(dinfo)), dinfo['cond_sex_col'])):
    val_donors.append(list(dinfo.iloc[val_idx].index.values))
val_masks = []
for val_donor in val_donors:
    val_masks.append(adata.obs[subid_col].isin(val_donor).values)

#%%
###############################################################################
# prep
###############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch_geometric

"""
x = data.x.to('cuda')
edge_index = data.edge_index.to('cuda')
xx = model.conv1(x, edge_index)
"""

class MLP(torch.nn.Module):
    r""" A customary MLP module.
    
    """
    def __init__(
        self, 
        channels: list,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()

        self.channels = channels
        if len(channels) < 2:
            raise ValueError("The list of dimensions must contain at least two values.")
        
        self.layers = torch.nn.Sequential()
        for i in range(len(channels) - 1):
            self.layers.append(
                torch.nn.Linear(channels[i], channels[i + 1],)
            )
            if i < len(channels) - 2:
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = 96, 
                 num_class: int = None,
                 heads: int = 4, 
                 drop_rate: Optional[float] = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class
        self.heads = heads
        self.drop_rate = drop_rate

        self.conv1 = torch_geometric.nn.GCNConv(in_channels=in_channels,
                                                out_channels=out_channels)
        self.conv2 = torch_geometric.nn.GCNConv(in_channels=out_channels,
                                                out_channels=out_channels)
        self.conv3 = torch_geometric.nn.GCNConv(in_channels=out_channels,
                                                out_channels=num_class)
        self.layers = torch.nn.ModuleList([self.conv1, self.conv2, self.conv3])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.elu(x)
        return x

    def predict(self, data, beta=1):
        r"""
        
        """
        self.eval()
        self.to('cpu')
        with torch.no_grad():
            x = self(data)
        x = torch.nn.functional.softmax(x, dim=1).detach().numpy()
        pred_score = (-1)*x[:, 0]**beta + (0)*x[:,1]**beta + (1)*x[:, 2]**beta  
        return pred_score.flatten()
    
def create_model(
    model: str,
    n_features: int,
    # data_loader: torch.utils.data.DataLoader,
):
    if model == 'GAT':
        model = PASCode.model.GAT(
            in_channels=n_features, out_channels=64, num_class=3, heads=4)
    if model == 'GCN':
        model = GCN(in_channels=n_features, out_channels=64, num_class=3)
    if model == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    if model == 'MLP':
        model = MLP(channels=[n_features, 64, 3])
    return model

def prep_data(
    model: str,
    adata: sc.AnnData
):
    if model == 'GAT' or model == 'GCN':
        data = PASCode.model.Data().adata2gdata(
            adata, 
            y=adata.obs['rra_pac'].values + 1, 
            trn_mask=adata.obs['train_mask'].values, 
            val_mask=adata.obs['val_mask'].values)
        data_loader = PASCode.model.Data().gdata2batch(
            data, 
            batch_size=16, # NOTE
            num_parts=64,  # NOTE
            shuffle=True)
        return data, data_loader
    if model =='MLP':
        ## torch data loader
        X = adata.X[adata.obs['train_mask'].values]
        y = adata.obs['rra_pac'].values[adata.obs['train_mask'].values]
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float32).to('cuda'), 
                torch.tensor(y + 1, dtype=torch.long).to('cuda')), # NOTE adding 1 to [-1,0,1] to avoid "CUDA error: device-side assert triggered"
            batch_size=128, shuffle=True)
        return data_loader
    if model == 'RF':
        X = adata.X
        y = adata.obs['rra_pac'].values
        return X, y

def run_model(
    model,
    model_name: str,
    # NOTE options
    X_trn: Optional[np.ndarray],
    y_trn: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    data_loader: Optional[torch.utils.data.DataLoader],
    data: Optional[torch_geometric.data.Data],
):
    if model_name == 'GAT' or model_name == 'GCN':
        best_model = PASCode.model.Trainer(model=model, device='cuda').train(
            trn_data_loader=data_loader, data_val=data, val_data_loader=None,
            max_epoch=100, lr=1e-3, lr_decay=[2, 0.5], early_stopping=10, weight_decay=1e-3)
        model = best_model
        model.eval()
        model.to('cpu')
        with torch.no_grad():
            x = model(data)
        x = x[data.val_mask] # NOTE
        x = torch.nn.functional.softmax(x, dim=1).detach().numpy()
        y_pred = np.argmax(x, axis=1)
        s_pred = x[:, 2] - x[:, 0]
    
    if model_name == 'RF':
        model.fit(X_trn, y_trn)
        y_pred = model.predict(X_val) + 1 #NOTE
        assert all(model.classes_ == np.array([-1,0,1])), "model.classes_ is not in the order [-1, 0, 1]"
        prob = model.predict_proba(X_val)
        s_pred =  prob[:,2] - prob[:,0]

    if model_name == 'MLP':
        model = model.to('cuda')
        X_val = torch.tensor(X_val, dtype=torch.float32).to('cuda')
        y_val = torch.tensor(y_val + 1, dtype=torch.long).to('cuda') # NOTE adding 1 to [-1,0,1] to avoid "CUDA error: device-side assert triggered"

        patience = 10
        early_stopping = 0
        min_loss = np.inf

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2)
        
        val_losses = []
        for epoch in range(100):
            trn_loss = 0
            for x, y in data_loader:
                x, y = x.to('cuda'), y.to('cuda')
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                trn_loss += loss.item()
            trn_loss = trn_loss / len(data_loader)
            model.eval()
            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val)

            print(f"epoch: {epoch}, trn loss: {trn_loss}, val loss: {val_loss}, early stopping: {early_stopping}")
            if scheduler:
                scheduler.step(val_loss)
            if val_loss >= min_loss:
                early_stopping += 1
            else:
                min_loss = val_loss
                early_stopping = 0
            if early_stopping > patience:
                break
                
            val_losses.append(val_loss)
        
        model.eval()
        model.to('cpu')
        X_val = X_val.to('cpu')
        with torch.no_grad():
            x = model(X_val)
        x = torch.nn.functional.softmax(x, dim=1).detach().numpy()
        y_pred = np.argmax(x, axis=1)
        s_pred = x[:, 2] - x[:, 0]

    return y_pred - 1, s_pred # NOTE subtract 1

#%%
import sklearn, scipy
model_names = ['GAT', 'GCN', 'MLP', 'RF']
metrics = ['f1-score', 'precision', 'recall', 'mse', 'pearson']

res = pd.DataFrame(
    columns = [model+'_'+metric for model in model_names for metric in metrics],
    index = [f'cross_val_{i}' for i in range(len(val_masks))]
)

for i, val_mask in enumerate(val_masks):
    print(f"\n ====================== Cross val {i} ====================== \n")

    adata.obs['train_mask'] = ~val_mask
    adata.obs['val_mask'] = val_mask

    for model_name in model_names:
        print(f"\n ---------------- {model_name} ---------------- \n")

        model = create_model(model=model_name, n_features=adata.X.shape[1])

        data = None
        data_loader = None
        X_trn = adata.X[adata.obs['train_mask'].values]
        y_trn = adata.obs['rra_pac'].values[adata.obs['train_mask'].values]
        X_val = adata.X[adata.obs['val_mask'].values]
        y_val = adata.obs['rra_pac'].values[adata.obs['val_mask'].values]

        if model_name == 'GAT' or model_name == 'GCN':
            data, data_loader = prep_data(model=model_name, adata=adata)
        if model_name == 'MLP':
            data_loader = prep_data(model=model_name, adata=adata) # trn
            X_val = adata.X[adata.obs['val_mask'].values]
            y_val = adata.obs['rra_pac'].values[adata.obs['val_mask'].values]
        if model_name == 'RF':
            X, y = prep_data(model=model_name, adata=adata)
            X_trn = X[adata.obs['train_mask'].values]
            y_trn = y[adata.obs['train_mask'].values]
            X_val = X[adata.obs['val_mask'].values]
            y_val = y[adata.obs['val_mask'].values]

        y_pred, s_pred = run_model(
            model=model, 
            model_name=model_name,
            data=data,
            data_loader=data_loader,
            X_trn=X_trn,
            y_trn=y_trn,
            X_val=X_val,
            y_val=y_val)

        ## pred res
        y_true = y_val
        res.loc[f'cross_val_{i}', model_name+'_f1-score'] = classification_report(
            y_true, y_pred, output_dict=True)['macro avg']['f1-score']
        res.loc[f'cross_val_{i}', model_name+'_precision'] = classification_report(
            y_true, y_pred, output_dict=True)['macro avg']['precision']
        res.loc[f'cross_val_{i}', model_name+'_recall'] = classification_report(
            y_true, y_pred, output_dict=True)['macro avg']['recall']
        res.loc[f'cross_val_{i}', model_name+'_mse'] = sklearn.metrics.mean_squared_error(
            y_true, s_pred)
        res.loc[f'cross_val_{i}', model_name+'_pearson'] = scipy.stats.pearsonr(
            y_true, s_pred)[0]

res.to_csv('/home/che82/athan/PASCode/code/github_repo/figures/s1/model_benchmarking_res.csv')

#%%
###############################################################################
# plot results
###############################################################################
res = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/figures/s1/model_benchmarking_res.csv', index_col=0)

custom_palette = {
    'MLP':  '#1f77b4', 
    'GCN':  '#ff7f0e', 
    'RF':   '#2ca02c',  
    'GAT': '#d62728', # red
}

import seaborn as sns
import matplotlib.patches as mpatches

num_folds = 6

Metrics = ['F1-score', 'Precision', 'Recall', 'MSE', 'Pearson']

fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=3)

for idx, (metric, Metric) in enumerate(zip(metrics, Metrics)):
    row, col = idx // (2+1), idx % 3
    
    axes[row, col].set_title(Metric, fontsize=17)
    axes[row, col].grid()

    # plt.boxplot(
    #     [[res.loc[f'cross_val_{i}', model_name+'_'+metric] for i in range(len(val_masks))] for model_name in model_names],
    #     labels=model_names)
    sns.boxplot(
        y=[res.loc[f'cross_val_{i}', model_name+'_'+metric] for model_name in model_names for i in range(num_folds)],
        x=[model_name for model_name in model_names for i in range(num_folds)],
        ax=axes[row, col],
        palette=custom_palette
    )
    sns.swarmplot(
        y=[res.loc[f'cross_val_{i}', model_name+'_'+metric] for model_name in model_names for i in range(num_folds)],
        x=[model_name for model_name in model_names for i in range(num_folds)],
        ax=axes[row, col],
        color='black',
        size=5
    )

    for tick in axes[row, col].get_xticks():
        axes[row, col].axvline(
            x=tick,
            color='grey',
            linestyle='-',
            linewidth=0.5,
            alpha=0.7)
            
patches = [mpatches.Patch(color=color, label=label) for label, color in custom_palette.items()]

fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.05),
        ncol=len(patches), fontsize=25)

plt.savefig(f'./model_benchmarking.pdf', dpi=600)
plt.close()
# plt.show()

                                                                                                                                                                                                # %%
