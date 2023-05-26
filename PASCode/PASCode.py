from .utils import *

import numpy as np
from scipy.spatial import KDTree

class PASCode:
    r"""
    Args:
        sampleid_name ([str]): [sample id column name]
        phenotype_name ([str]): [phenotype column name]
        pos_phenotype_name ([str]): [positive phenotype name]
    """
    def __init__(self, sampleid_name, phenotype_name, pos_phenotype_name):
        self.sampleid_name = sampleid_name
        self.phenotype_name = phenotype_name
        self.pos_phenotype_name = pos_phenotype_name

    def fit(self, adata, fdr_thres):
        pac_milo = run_milo(adata, return_pac=True, 
                            sampleid_name=self.sampleid_name, 
                            phenotype_name=self.phenotype_name,
                            fdr_thres=fdr_thres)
        self.scacc, pac_scacc = run_scacc(X=adata.X, 
                                          meta=adata.obs, 
                                          return_pac=True, 
                                          sampleid_name=self.sampleid_name, 
                                          phenotype_name=self.phenotype_name, 
                                          pos_phenotype_name=self.pos_phenotype_name,
                                          fdr_thres=fdr_thres)
        # self.anchor_pac = np.intersect1d(pac_milo, pac_scacc)
        self.anchor_pac = pac_milo
        self.anchor_pac_index = np.where(np.isin(adata.obs.index.values, self.anchor_pac))[0]
        z = self.scacc.get_latent_space(adata.X)
        self.tree = KDTree(z)

    def predict(self, x, cell_names=None):
        r"""
        For every point in testing data, find its nearest neighbor among all points in training data, if this nn is in pacs_train then this testing point is pac. 
        time complexity for plain implementation: O(n_test*n_train) (think of it as O(N^2)) --> too slow ### NOTE euc dist computation is O(D) where D is #dim, so for 3D points its just O(1). 
        use some fast algorithm (if there is any) to find nearest neighbor.
        KD-tree: O(Nlog(N)) (Octree is a similar method except it is for 3D points in particular)
        
        Args:

        Returns:

        """
        z = self.scacc.get_latent_space(x)
        test_point_is_pac = np.zeros(z.shape[0])
        for i, query_point in enumerate(z):
            _, index = self.tree.query(query_point)
            if index in self.anchor_pac_index:
                test_point_is_pac[i] = int(1)
        if cell_names is None:
            return  np.where(test_point_is_pac == 1)[0]
        else:
            return cell_names[np.where(test_point_is_pac == 1)[0]]

    def get_latent_space(self, x):
        return self.scacc.get_latent_space(x)

    def get_cluster(self, x):
        return self.scacc.get_cluster(x)
