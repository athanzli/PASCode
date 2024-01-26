from .random_seed import *
set_seed(RAND_SEED)

import copy
import torch
import torch_geometric
import pandas as pd
from typing import Optional, List, Union
import anndata

class GAT(torch.nn.Module):
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

        self.conv1 = torch_geometric.nn.GATConv(in_channels=in_channels,
                                                out_channels=out_channels,
                                                heads=heads, concat=True, 
                                                drop_rate=drop_rate)
        self.conv2 = torch_geometric.nn.GATConv(in_channels=out_channels*heads,
                                                out_channels=out_channels,
                                                heads=heads, concat=True, 
                                                drop_rate=drop_rate)
        self.conv3 = torch_geometric.nn.GATConv(in_channels=out_channels*heads,
                                                out_channels=num_class,
                                                heads=heads, concat=False, 
                                                drop_rate=drop_rate)
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

        # Convert adjacency matrix to COO format and then to edge indices
        coo_data = adata.obsp['connectivities'].tocoo()
        edge_index = torch.LongTensor([coo_data.row, coo_data.col])

        # Create the PyTorch Geometric Data object
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

        print('Training...')
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
        
        self._plot_learning_curve(epoch_list, train_losses, val_losses)

        return best_model

