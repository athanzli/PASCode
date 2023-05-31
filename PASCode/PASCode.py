from .utils import *

import numpy as np
from scipy.spatial import KDTree

class PASCode:
    r"""
    Args:
        sampleid_name (str): sample id column name.
        phenotype_name (str): phenotype column name
        pos_phenotype_name (str): positive phenotype name.
    """
    def __init__(self, sampleid_name, phenotype_name, pos_phenotype_name):
        self.sampleid_name = sampleid_name
        self.phenotype_name = phenotype_name
        self.pos_phenotype_name = pos_phenotype_name

    def fit(self, 
            adata, 
            require_pac_pos_neg=False,
            fdr_thres=.05, 
            n_neighbors=15, 
            n_pcs=50, 
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
            milo_prop=0.1,):
        r"""
        Args:
            adata (AnnData): AnnData object.
            require_pac_pos_neg (bool): whether to return positive and negative pacs.
            fdr_thres (float): fdr threshold.
            n_neighbors (int): number of neighbors.
            n_pcs (int): number of pcs.
            latent_dim (int): latent dimension.
            n_clusters (int): number of clusters.
            lambda_cluster (float): lambda cluster.
            lambda_phenotype (float): lambda phenotype.
            device (str): device.
            epoch_pretrain (int): epoch pretrain.
            epoch_train (int): epoch train.
            batch_size (int): batch size.
            lr_pretrain (float): learning rate pretrain.
            lr_train (float): learning rate train.

        Returns:
            numpy.ndarray: anchor pacs.
            OR.
            numpy.ndarray, numpy.ndarray: anchor pacs positive, anchor pacs negative.
        """
        pac_milo, _ = run_milo(adata, 
                            return_pac_pos_neg=require_pac_pos_neg,
                            sampleid_name=self.sampleid_name, 
                            phenotype_name=self.phenotype_name,
                            fdr_thres=fdr_thres,
                            n_neighbors=n_neighbors, 
                            n_pcs=n_pcs,
                            make_nhoods_prop=milo_prop)
        self.pac_milo_pos = 0 if type(_) != np.ndarray else pac_milo
        self.pac_milo_neg = 0 if type(_) != np.ndarray else _
        self.scacc = run_scacc(X=adata.X, 
                            meta=adata.obs, 
                            sampleid_name=self.sampleid_name, 
                            phenotype_name=self.phenotype_name, 
                            pos_phenotype_name=self.pos_phenotype_name,
                            fdr_thres=fdr_thres,
                            latent_dim=latent_dim, 
                            n_clusters=n_clusters, 
                            lambda_cluster=lambda_cluster, 
                            lambda_phenotype=lambda_phenotype, 
                            device=device,
                            epoch_pretrain=epoch_pretrain,
                            epoch_train=epoch_train,
                            batch_size=batch_size,
                            lr_pretrain=lr_pretrain,
                            lr_train=lr_train,)
        z = self.get_latent_space(adata.X)
        self.tree = KDTree(z)
        if type(self.pac_milo_pos) != np.ndarray:
            self.anchor_pac_bool = pac_milo & 1
            self.anchor_pac = adata.obs.index[self.anchor_pac_bool].values # NOTE: self.anchor_pac are pac names
            self.anchor_pac_ind = np.where(1 == self.anchor_pac_bool)[0]
            return
        self.anchor_pac_pos_bool = (self.pac_milo_pos & 1).astype(bool)
        self.anchor_pac_neg_bool = (self.pac_milo_neg & 1).astype(bool)
        self.anchor_pac_pos = adata.obs.index[self.anchor_pac_pos_bool].values # NOTE pac names
        self.anchor_pac_neg = adata.obs.index[self.anchor_pac_neg_bool].values # NOTE pac names
        self.anchor_pac_pos_ind = np.where(self.anchor_pac_pos_bool == 1)[0]
        self.anchor_pac_neg_ind = np.where(self.anchor_pac_neg_bool == 1)[0]

    def predict(self, adata, return_pac_pos_neg=False, k=1, vote_thres=1):
        r"""
        For every point in testing data, find its nearest neighbor among all points in training data, if this nn is in pacs_train then this testing point is pac. 
        time complexity for plain implementation: O(n_test*n_train) (think of it as O(N^2)) --> too slow ### NOTE euc dist computation is O(D) where D is #dim, so for 3D points its just O(1). 
        use some fast algorithm (if there is any) to find nearest neighbor.
        KD-tree: O(Nlog(N)) (Octree is a similar method except it is for 3D points in particular)
        
        Args:
            arg1 (type): Description of the first argument.
            arg2 (type): Description of the second argument.
            
        Returns:
            type: Description of the return value.
        """
        if not return_pac_pos_neg:
            z = self.get_latent_space(adata.X)
            _, indices = self.tree.query(z, k=k) # query all points at once
            indices = indices.reshape(-1,1)
            vote_counts = self.anchor_pac_bool[indices].sum(axis=1)
            test_point_is_pac = (vote_counts >= k*vote_thres).astype(int)
            return test_point_is_pac == 1

        z = self.get_latent_space(adata.X)
        _, indices = self.tree.query(z, k=k) # query all points at once
        indices = indices.reshape(-1,1)
        vote_counts = self.anchor_pac_pos_bool[indices].sum(axis=1)
        test_point_is_pac_pos = (vote_counts >= k*vote_thres).astype(int)
        vote_counts = self.anchor_pac_neg_bool[indices].sum(axis=1)
        test_point_is_pac_neg = (vote_counts >= k*vote_thres).astype(int)
        return (test_point_is_pac_pos == 1), (test_point_is_pac_neg == 1)

    def get_latent_space(self, x):
        return self.scacc.get_latent_space(x)

    def get_cluster(self, x):
        return self.scacc.get_cluster(x)
    
