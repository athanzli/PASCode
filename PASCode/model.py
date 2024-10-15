from .random_seed import *
set_seed(RAND_SEED)

import copy
import torch
import torch_geometric
import scanpy as sc
from typing import Mapping, Optional, List, Union, Any
import anndata
import time

from .graph import build_graph
from .utils import subject_info, subsample_donors
from .da import agglabel

class GAT(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = 64, 
                 num_class: int = 3,
                 heads: int = 4, 
                 drop_rate: Optional[float] = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class
        self.heads = heads
        self.drop_rate = drop_rate

        self.conv1 = torch_geometric.nn.GATConv(in_channels=in_channels,
                                                out_channels=out_channels,
                                                heads=heads, concat=True, 
                                                dropout=drop_rate)
        self.conv2 = torch_geometric.nn.GATConv(in_channels=out_channels*heads,
                                                out_channels=out_channels,
                                                heads=heads, concat=True, 
                                                dropout=drop_rate)
        self.conv3 = torch_geometric.nn.GATConv(in_channels=out_channels*heads,
                                                out_channels=num_class,
                                                heads=heads, concat=False, 
                                                dropout=drop_rate)
        self.layers = torch.nn.ModuleList([self.conv1, self.conv2, self.conv3])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.elu(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def predict(self, data, device='cpu'):
        r"""
        Predicts PAC scores for the given data.

        Args:
            - data: PyTorch Geometric Data object.
            - device: Device type (e.g., 'cpu', 'cuda').
        """
        if not isinstance(data, torch_geometric.data.Data):
            print("Building PyTorch Geometric Data...")
            data = Data().adata2gdata(data)
        self.eval()
        self.to(device)
        with torch.no_grad():
            x = self(data)
        x = torch.nn.functional.softmax(x, dim=1).detach().numpy()
        pred_score = x[:, 2] - x[:, 0]
        return pred_score.flatten()

class Data:
    def __init__(self):
        pass
    
    @staticmethod
    def adata2gdata(adata: anndata.AnnData, 
                    y: Optional[np.ndarray] = None, 
                    trn_mask: Optional[np.ndarray] = None, 
                    val_mask: Optional[np.ndarray] = None) -> torch_geometric.data.Data:
        """
        Converts an AnnData object to a PyTorch Geometric Data object.

        Parameters:
        - adata (anndata.AnnData): The input AnnData object that contains the data matrix and the graph connectivity.
        - y (np.ndarray or None): Target labels array. If None, 'y' will not be included in the returned data object.
        - trn_mask (np.ndarray, optional): A boolean mask indicating which samples are used for training. 
                                          If None, 'train_mask' will not be included in the returned data object.
        - val_mask (np.ndarray, optional): A boolean mask indicating which samples are used for validation. 
                                          If None, 'val_mask' will not be included in the returned data object.

        Returns:
        - data (torch_geometric.data.Data): A PyTorch Geometric Data object.
        """

        if 'connectivities' not in adata.obsp:
            print("No graph connectivities found in adata.obsp.")
            build_graph(adata, run_umap=False)

        # convert connectivies to COO format and then to edge indices
        coo_data = adata.obsp['connectivities'].tocoo()
        edge_index = torch.LongTensor([coo_data.row, coo_data.col])

        data = torch_geometric.data.Data(
            x=torch.from_numpy(adata.X).to(torch.float32),
            y=torch.from_numpy(y).to(torch.long) if y is not None else None,
            edge_index=edge_index,
            train_mask=torch.from_numpy(trn_mask) if trn_mask is not None else None,
            val_mask=torch.from_numpy(val_mask) if val_mask is not None else None,
            idx=torch.from_numpy(np.arange(adata.shape[0]))
        ).to('cpu')
    
        return data
    
    @staticmethod
    def gdata2batch(data: torch_geometric.data.Data, 
                    batch_size: int = 128, 
                    num_parts: int = 128*16, 
                    shuffle: bool = True) -> torch_geometric.loader.ClusterLoader:
        """
        Constructs batches from the given PyTorch Geometric Data object using ClusterGCN (https://arxiv.org/abs/1905.07953).

        Parameters:
        - data (torch_geometric.data.Data): The input PyTorch Geometric Data object.
        - batch_size (int, optional): The batch size. Default is 128.
        - num_parts (int, optional): The number of parts for clustering. Default is 128*16.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
        - data_loader (torch_geometric.loader.ClusterLoader): A PyTorch Geometric ClusterLoader containing the batches.
        """
        print('Constructing batches...')
        # NOTE https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-7970609
        clusterdata = torch_geometric.loader.ClusterData(
            data, num_parts=num_parts)
        data_loader = torch_geometric.loader.ClusterLoader(
            clusterdata, batch_size=batch_size, shuffle=shuffle)
        print("Batch construction done.")
        return data_loader

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 device: Union[str, torch.device] = 'cpu') -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)

    def _initialize_optimizer(
            self, 
            lr: float, 
            weight_decay: float, 
            lr_decay: Optional[float] = None):            
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_decay[1], patience=lr_decay[0]) if lr_decay else None
        return optimizer, scheduler
    
    def _train_one_epoch(
            self, 
            trn_data_loader: torch_geometric.loader.ClusterLoader,
            criterion: torch.nn.Module, 
            optimizer: torch.optim.Optimizer) -> float:
        criterion.to(self.device)
        total_loss = 0
        self.model.train()
        for batch in trn_data_loader:
            batch.to(self.device)
            optimizer.zero_grad()
            out = self.model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(trn_data_loader)
    
    def _val_one_epoch(
            self, 
            data_val: torch.Tensor, 
            val_data_loader: torch_geometric.loader.ClusterLoader,
            criterion: torch.nn.Module) -> float:
        self.model.eval()
        self.model.to('cpu')
        criterion.to('cpu')
        with torch.no_grad():
            if val_data_loader is None:
                out = self.model(data_val)
            else:
                out = torch.zeros(data_val.num_nodes, self.model.num_class, device='cpu')
                for batch in val_data_loader:
                    out[batch.idx] = self.model(batch)
            loss = criterion(out[data_val.val_mask], data_val.y[data_val.val_mask])
        self.model.to(self.device)
        return loss.item()

    @staticmethod
    def _plot_learning_curve(
            epoch_list: List[int], 
            train_losses: List[float], 
            val_losses: Optional[List[float]]):
        plt.plot(epoch_list, train_losses)
        legends = ['train']
        if val_losses:
            plt.plot(epoch_list, val_losses)
            legends.append('val')
        plt.legend(legends, loc='lower left')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('loss')
        plt.show()

    def train(self,
              trn_data_loader: torch_geometric.loader.ClusterLoader,
              data_val: Optional[torch.Tensor] = None, 
              val_data_loader: Optional[torch_geometric.loader.ClusterLoader] = None, 
              max_epoch: int = 100, 
              lr: float = 1e-3,
              lr_decay: Optional[List[float]] = [2, 0.5],
              weight_decay: Optional[float] = 0,
              early_stopping: Optional[int] = None,
              class_weight: List[float] = [1, 1, 1],
              plot_training_curve: bool = True,
              print_epoch_interval: Optional[int] = 1) -> torch.nn.Module:
        """
        Trains a given PyTorch model using the provided data loaders and training parameters.
        
        Parameters:
        - model: A PyTorch module representing the neural network model.
        - trn_data_loader: Training data loader.
        - data_val: Optional validation data tensor.
        - val_data_loader: Optional validation data loader.
        - max_epoch: Maximum training epochs.
        - lr: Learning rate.s
        - weight_decay: Weight decay for optimizer.
        - early_stopping: Number of epochs for early stopping.
        - device: Device type (e.g., 'cpu', 'cuda').
        - class_weight: Class weights for the loss function.
        
        Returns:
        - Trained model (best model if validation data is provided, otherwise the model at the last epoch).
        """
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(self.model.state_dict())
        
        optimizer, scheduler = self._initialize_optimizer(lr, weight_decay=weight_decay, lr_decay=lr_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32))

        epoch_count, min_loss = 0, float('inf')
        train_losses, val_losses, epoch_list = [], [], []

        print('\n============================= Training GAT... =============================')
        st = time.time()
        for epoch in range(max_epoch):
            epoch_list.append(epoch + 1)
            train_loss = self._train_one_epoch(trn_data_loader, criterion, optimizer)
            train_losses.append(train_loss)

            if (epoch + 1) % print_epoch_interval == 0:
                print(
                    f"Epoch: {epoch + 1}/{max_epoch} - "
                    f"lr: {optimizer.param_groups[0]['lr']:.3e} - "
                    f"train_loss: {train_loss:.3f}", end=' ')
            
            val_loss = None
            if data_val:
                val_loss = self._val_one_epoch(data_val, val_data_loader, criterion)
                val_losses.append(val_loss)
                if (epoch + 1) % print_epoch_interval == 0:
                    print(f"- val_loss: {val_loss:.3f}", end=' ')
            
            latest_loss = val_loss if val_loss else train_loss
            if scheduler:
                scheduler.step(latest_loss)
            if early_stopping:
                if latest_loss >= min_loss:
                    epoch_count += 1
                else:
                    min_loss = latest_loss
                    best_model.load_state_dict(self.model.state_dict())
                    epoch_count = 0
                if epoch_count > early_stopping:
                    break
            print()
        print(f"\n============================= Training Time cost (s): {(time.time() - st):.2f} =============================")
        
        if plot_training_curve:
            self._plot_learning_curve(epoch_list, train_losses, val_losses)

        return best_model

def get_val_mask(
    adata: sc.AnnData,
    subid_col: str,
    cond_col: str,
    pos_cond: str,
    neg_cond: str,
    aggreg_label_col: str = 'aggreg_label',
    sex_col: str = None,
    mode: str = 'cell',
    val_percent: float = 0.1
):
    r"""
    Get the validation mask for the given AnnData object.
    Make sure 'aggre_label' is in adata.obs.

    Args:
        - adata: AnnData object.
        - subid_col: Column name of the subject ID.
        - cond_col: Column name of the condition.
        - pos_cond: Positive condition.
        - neg_cond: Negative condition
        - sex_col: Column name of sex.
        - mode: 'donor' or 'cell'. If donor, the validation mask will be created based on the donors.
                If cell, the validation mask will be created based on the cells.
        - val_percent: The percentage of validation samples. Default is 0.1.
    """
    if mode == 'donor':
        if sex_col is not None:
            assert adata.obs[sex_col].nunique() == 2
            dinfo = subject_info(
                adata.obs, 
                subid_col=subid_col, 
                columns=[cond_col, 'Sex'])
            num = (dinfo.shape[0] // (int(100*val_percent)*4))
            if num == 0:
                raise ValueError("The number of subjects is too small for subsampling. Try cell mode.")
            mp = dinfo[(dinfo['Sex'] == 'Male') & (dinfo[cond_col] == pos_cond)] \
                    .sample(n=num).index.values
            fp = dinfo[(dinfo['Sex'] == 'Female') & (dinfo[cond_col] == pos_cond)] \
                    .sample(n=num).index.values
            mn = dinfo[(dinfo['Sex'] == 'Male') & (dinfo[cond_col] == neg_cond)] \
                    .sample(n=num).index.values
            fn = dinfo[(dinfo['Sex'] == 'Female') & (dinfo[cond_col] == neg_cond)] \
                    .sample(n=num).index.values
            sub_sel = np.concatenate([mp, fp, mn, fn])
        else:
            dinfo = subject_info(
                adata.obs, 
                subid_col=subid_col, 
                columns=[cond_col])
            num = (dinfo.shape[0] // (int(100*val_percent)*2))
            if num == 0:
                raise ValueError("The number of subjects is too small for subsampling. Try cell mode.")
            psel = dinfo[dinfo[cond_col] == pos_cond] \
                    .sample(n=num).index.values
            nsel = dinfo[dinfo[cond_col] == neg_cond] \
                    .sample(n=num).index.values
            sub_sel = np.concatenate([psel, nsel])
        adata.obs['val_mask'] = adata.obs[subid_col].isin(sub_sel).values
        adata.obs['train_mask'] = ~adata.obs['val_mask']

    elif mode == 'cell':
        if aggreg_label_col is None:
            raise ValueError("aggreg_label_col should be provided when mode is 'cell'.")
        nums = adata.obs[aggreg_label_col].value_counts()
        sel1 = adata.obs[adata.obs[aggreg_label_col]==0].sample(n=nums[0]//int(100*val_percent)).index
        sel2 = adata.obs[adata.obs[aggreg_label_col]==-1].sample(n=nums[-1]//int(100*val_percent)).index
        sel3 = adata.obs[adata.obs[aggreg_label_col]==1].sample(n=nums[1]//int(100*val_percent)).index
        sel = np.concatenate([sel1, sel2, sel3])
        adata.obs['val_mask'] = adata.obs.index.isin(sel)
        adata.obs['train_mask'] = ~adata.obs['val_mask']

    else:
        raise ValueError("mode should be either 'donor' or 'cell'.")

    return adata.obs['val_mask']

def train_model(
    adata: sc.AnnData,
    agglabel_col: str,
    device: Union[str, torch.device] = 'cpu',
    batch_size: int = 128,
    num_parts: int = 2048,
    plot_training_curve: bool = False,
    save: bool = True,
    save_path: str = './'
):
    #
    data = Data().adata2gdata(
        adata,
        y=adata.obs[agglabel_col].values + 1, # NOTE
        trn_mask=adata.obs['train_mask'].values,
        val_mask=adata.obs['val_mask'].values)
    data_loader = Data().gdata2batch(
        data,
        batch_size=batch_size,
        num_parts=num_parts,
        shuffle=True)
    #
    model = GAT(
        in_channels=adata.X.shape[1], out_channels=64, num_class=3, heads=4)

    best_model = Trainer(model=model, device=device).train(
        trn_data_loader=data_loader, data_val=data, val_data_loader=None,
        max_epoch=100, lr=1e-3, lr_decay=[2, 0.5], early_stopping=10, weight_decay=1e-3, # NOTE
        class_weight=[1, 1, 1], plot_training_curve=plot_training_curve)
    model = best_model

    if save:
        torch.save(model.state_dict(), './trained_model.pt')

    return model


def custom_train(
    adata: sc.AnnData,
    subid_col: str,
    cond_col: str,
    pos_cond: str,
    neg_cond: str,
    sex_col: str = None,
    subsample: bool = True,
    subsample_num: str = None,
    subsample_mode: str = 'top',
    da_methods: List[str] = ['milo', 'meld', 'daseq'],
    device: Union[str, torch.device] = 'cpu',
    val_mask_mode: str = 'cell',
    graph_build_use_rep: str = 'X_pca',
    batch_size: int = 128,
    num_parts: int = 2048
):
    r"""
    Training from sratch with custom options, including running DA and RRA.

    Args:
        - adata: AnnData object.
        - subid_col: Column name of the subject ID.
        - cond_col: Column name of the condition.
        - pos_cond: Positive condition.
        - neg_cond: Negative condition.
        - sex_col: Column name of sex.
        - subsample_mode: 'top' or 'random'.
        - da_methods: List of DA methods to use.
        - val_mask_mode: 'donor' or 'cell'.
        - batch_size: Batch size for training.
        - graph_build_use_rep: Representation to use for building the graph. Default is 'X_pca'.
        - num_parts: Number of parts for clustering.
    """

    if subsample:
        sc.pp.scale(adata) # NOTE normally we expect the data to have already been scaled, but in case it's not, we scale it here before running PCA. If already scaled, this will have no effect.
        build_graph(adata, use_rep=graph_build_use_rep)
        ## subsample data
        if subsample_num is None:
            dinfo = subject_info(adata.obs, subid_col, columns=[cond_col])
            min_num = dinfo[cond_col].value_counts().min()
            subsample_num = f"{min_num}:{min_num}"
        adata_sub = subsample_donors(
            adata=adata,
            subsample_num=subsample_num,
            subid_col=subid_col,
            cond_col=cond_col,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            sex_col=sex_col,
            mode=subsample_mode,
        )
        adata.obs['subsampled'] = adata.obs.index.isin(adata_sub.obs.index)
        ## build graph
        build_graph(adata=adata_sub, use_rep=graph_build_use_rep)
        ## run DA to get aggregated labels
        agglabel(
            adata_sub,
            subid_col,
            cond_col,
            pos_cond,
            neg_cond,
            da_methods=da_methods
        )
        ## assign DA res and agg. labels back to adata
        for method in da_methods:
            adata.obs.loc[adata_sub.obs.index, f'subsampled_{method}'] = adata_sub.obs[method].values
        adata.obs.loc[adata_sub.obs.index, 'subsampled_aggreg_label'] = adata_sub.obs['aggreg_label'].values
        ## get val_mask
        get_val_mask(
            adata=adata_sub,
            subid_col=subid_col,
            cond_col=cond_col,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            aggreg_label_col='aggreg_label',
            sex_col=sex_col,
            mode=val_mask_mode
        )
        adata.obs['val_mask'] = adata.obs.index.isin(
            adata_sub.obs[adata_sub.obs['val_mask']].index)
        adata.obs['train_mask'] = adata.obs.index.isin(
            adata_sub.obs[adata_sub.obs['train_mask']].index)
        model = train_model(
            adata,
            agglabel_col='subsampled_aggreg_label',
            batch_size=batch_size,
            num_parts=num_parts,
            device=device
        )
    else:
        ## build graph
        sc.pp.scale(adata) # NOTE normally we expect the data to have already been scaled, but in case it's not, we scale it here before running PCA. If already scaled, this will have no effect.
        build_graph(adata=adata, use_rep=graph_build_use_rep)
        ## run DA to get aggregated labels
        agglabel(
            adata,
            subid_col,
            cond_col,
            pos_cond,
            neg_cond,
            da_methods=da_methods
        )
        ## get val_mask
        get_val_mask(
            adata=adata,
            subid_col=subid_col,
            cond_col=cond_col,
            pos_cond=pos_cond,
            neg_cond=neg_cond,
            aggreg_label_col='aggreg_label',
            sex_col=sex_col,
            mode=val_mask_mode
        )
        ## train GAT
        model = train_model(
            adata,
            agglabel_col='aggreg_label',
            batch_size=batch_size,
            num_parts=num_parts,
            device=device
        )

    return model


def score(
    adata: sc.AnnData,
    subid_col: str,
    cond_col: str,
    pos_cond: str,
    neg_cond: str,
    sex_col: str = None,
    subsample: bool = True,
    subsample_num: str = None,
    subsample_mode: str = 'top',
    da_methods: List[str] = ['milo', 'meld', 'daseq'],
    device: Union[str, torch.device] = 'cpu',
    val_mask_mode: str = 'cell',
    graph_build_use_rep: str = 'X_pca',
    batch_size: int = 128,
    num_parts: int = 2048        
):
    r"""
    Auto scoring function for PAC. 
    This automates every step from subsampling to training and scoring.
    Not recommended for users who want to control certain steps during the process (see tutorial for examples).

    Args:
        - adata: AnnData object.
        - subid_col: Column name of the subject ID.
        - cond_col: Column name of the condition.
        - pos_cond: Positive condition.
        - neg_cond: Negative condition
        - sex_col: Column of sex.
        - subsample_mode: 'top' or 'random'.
        - da_methods: List of DA methods to use.
        - val_mask_mode: 'donor' or 'cell'.
        - batch_size: Batch size for training.
        - graph_build_use_rep: Representation to use for building the graph. Default is 'X_pca'.
        - num_parts: Number of parts for clustering.
    """
    model = custom_train(
        adata=adata, 
        subid_col=subid_col,
        cond_col=cond_col,
        pos_cond=pos_cond,
        neg_cond=neg_cond,
        sex_col=sex_col,
        subsample=subsample,
        subsample_num=subsample_num,
        subsample_mode=subsample_mode,
        da_methods=da_methods,
        device=device,
        val_mask_mode=val_mask_mode,
        graph_build_use_rep=graph_build_use_rep,
        batch_size=batch_size,
        num_parts=num_parts
    )
    adata.obs['PAC_score'] = model.predict(adata)
    return adata.obs['PAC_score'].values

def load_model(
        state_dict_path: str,
        in_channels: int = 3401,
        out_channels: int = 64,
        num_class: int = 3,
        heads: int = 4,
        drop_rate: float = 0.0,
    ):
    state_dict = torch.load(state_dict_path)
    model = GAT(
        in_channels=in_channels,
        out_channels=out_channels,
        num_class=num_class,
        heads=heads,
        drop_rate=drop_rate)
    model.load_state_dict(state_dict, strict=True)
    return model


