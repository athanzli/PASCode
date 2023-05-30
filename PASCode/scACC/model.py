import sys
import gc
import torch
import sklearn.cluster
import sklearn.ensemble
import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from IPython.display import clear_output

import umap
import seaborn
from torch import nn
import torch.nn.functional as F

import os

# os.chdir('/home/che82/athan/pascode/github/PASCode/scACC/')

from .utils import *

###############################################################################
################################ The scACC Model  ###########################
###############################################################################

class scACC():
    r"""
    The scACC model.

    Args:
        latent_dim: latent dimension
        n_clusters: number of clusters
        lambda_cluster: for kl-div loss
        lambda_phenotype: for entropy loss
        dropout: for AE
        device: ...
        alpha: ...
    """
    def __init__(self, 
            latent_dim=3, 
            n_clusters=30, 
            lambda_cluster=1, 
            lambda_phenotype=1, 
            device='cpu',
            alpha=1,
            dropout=.2
        ):
        
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.lambda_cluster = lambda_cluster
        self.lambda_phenotype = lambda_phenotype
        self.dropout = dropout
        self.alpha = alpha
        self.device = torch.device(device)

    # def __call__(self, x):
    #     r"""
    #     Returns:
    #         x_bar: reconstructed 
    #         q: clustering Q matrix 
    #         z: embedding
    #     """
    #     x = torch.tensor(x).float().to(self.device)
    #     z, x_bar = self.ae(x)
    #     q = calc_q(z, self.clusters, self.alpha)
    #     return x_bar, q, z

    class _AE(nn.Module):
        r"""
        Autoencoder module of scACC.
        """
        def __init__(self, latent_dim, input_dim=None, dropout=.2, device='cpu'):
            super().__init__()
            self.dropout = dropout
            self.input_dim = input_dim
            self.device = device
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, input_dim),
            )

        def forward(self, x):
            x = torch.tensor(x).float().to(self.device)
            z = self.encoder(x)
            x_bar = self.decoder(z)
            return x_bar
       

    def init_ae(self, X_train):
        r"""
        A helper function for creating the ae attribute in order for pretrained
        data loading.
        """
        self.ae = self._AE(
            input_dim=X_train.shape[1],
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            device=self.device).to(self.device)

    def train(self,
            X_train, 
            y_train, 
            epoch_pretrain=15,
            epoch_train=15, # do not train too many epochs, it may improve accuracy but will not make sense biologically
            batch_size=512,
            lr_pretrain=1e-3,
            lr_train=1e-3,

            require_pretrain_phase=True,
            require_train_phase=True, 
            evaluation=False,
            plot_evaluation=False, # plot metics per epoch
            id_train=None, X_test=None, y_test=None, id_test=None, # for printing out metrics per epoch
            fold_num=None,
        ):
        r"""
        Train scACC model, including pretraining phase and training phases

        Args:
            X_train: cell-by-genes data matrix. rows are cells, columns are genes,
                and entries are gene expression levels
            y_train: 
        """

        if batch_size*2 > X_train.shape[0]:
            print("Error: batch size too large.")
            sys.exit(1)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_train = torch.tensor(X_train, dtype=torch.float32) # without this line, X_train will still be torch.float64. why?
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        if X_test is not None:
            self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            self.X_test = torch.tensor(X_test, dtype=torch.float32)
        if y_test is not None:
            self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
            self.y_test = torch.tensor(y_test, dtype=torch.float32)

        self.evaluation = evaluation
        self.plot_evaluation = plot_evaluation

        self.id_train = id_train
        self.id_test = id_test

        self.epoch_train = epoch_train
        self.fold_num = fold_num
            
        if require_pretrain_phase:
            # print("Pretraining...")
            self._pretrain(X_train, lr=lr_pretrain, epoch_pretrain=epoch_pretrain, batch_size=batch_size)
            # print("Pretraining complete.\n")
        if require_train_phase:
            # print('Training...')
            self._train(X_train, y_train, lr=lr_train, epoch_train=epoch_train, batch_size=batch_size)
            # print("Training complete.\n")

    def _pretrain(self, X_train, lr, epoch_pretrain, batch_size, optimizer='adam'):
        r"""
        Pretraining phase.
        Train the AE module in scACC and initialize cluster self.. 
        """

        self.init_ae(X_train)

        train_loader = torch.utils.data.DataLoader(
            X_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        elif optimizer == '?': # TODO
            NotImplemented

        for epoch in range(epoch_pretrain):
            total_loss = 0
            for x in train_loader:
                x = x.to(self.device)
                x_bar  = self.ae(x)
                z = self.ae.encoder(x)
                optimizer.zero_grad()
                loss = F.mse_loss(x_bar, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print("epoch {}\t\t loss={:.4f}".format(epoch, total_loss/len(train_loader)))

        # print("Initializing cluster centroids...")
        self.clusters = torch.nn.Parameter(torch.Tensor(self.n_clusters, self.latent_dim))
        torch.nn.init.kaiming_normal_(self.clusters.data) # NOTE
        with torch.no_grad():
            z = self.ae.encoder(X_train)
        km = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=20)
        km.fit_predict(z.data.cpu().numpy())
        self.clusters.data = torch.tensor(km.cluster_centers_).to(self.device)

        # clean up space
        del z, x_bar, x, train_loader
        gc.collect()
        torch.cuda.empty_cache()


    def _train(self, X_train, y_train, lr, epoch_train, batch_size):
        r"""
        Training phase.
        """
        # prepare training data loader
        train_loader = torch.utils.data.DataLoader(
            subDataset(X_train, y_train), # from utils
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)

        # for eval
        self.accuracy_train = []
        self.roc_auc_train = []
        self.accuracy_test = []
        self.roc_auc_test = []
        self.loss_c = []
        self.loss_r = []
        self.loss_p = []
        self.loss_total = []
        self.epochs = []

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # train
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        for epoch in range(epoch_train):
            with torch.no_grad():
                z = self.ae.encoder(X_train)
            q = calc_q(z, self.clusters.data, self.alpha)
            p = target_distribution(q.data)
            self.P = p # for evaluation
            # minibatch gradient descent to train AE
            self.ae.train()
            for x, y, idx in train_loader:
                x = x.to(self.device)
                # x2 = add_noise(x)  # for denoising AE
                x_bar = self.ae(x)
                z = self.ae.encoder(x)
                q = calc_q(z, self.clusters.data, self.alpha)
                rec_loss = F.mse_loss(x_bar, x) # AE reconstruction loss
                kl_loss = F.kl_div(q.log(), p[idx]) # KL-div loss # NOTE do not use reduction='batchmean', it will significantly lower performance (why?)
                # phenotype entropy loss
                y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), dtype=torch.float32).to(self.device)
                wpheno = q.T@y + 1e-9 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
                wpheno /= wpheno.sum(dim=1, keepdim=True)               
                ent_loss = torch.mean(-1*torch.sum(wpheno*wpheno.log(), 1))
        
                # total joint loss
                loss = self.lambda_cluster*kl_loss + self.lambda_phenotype*ent_loss + rec_loss 

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                            

            if self.evaluation:
                self.evaluate(X_train, y_train, epoch)

    def get_latent_space(self, x):
        r"""
        Get the latent space of input data. 

        Args:
            x [numpy array]: input data

        Returns:
            latent space as a numpy array
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        z = self.ae.encoder(x)
        return z.detach().cpu().numpy()

    def plot_latent_space(self, X, label=None, reducer=None, 
                          title=None, require_distinguishable_colors=False):

        custom_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000", "#8000FF",
            "#FF007F", "#007FFF", "#7FFF00", "#FF7F00", "#00FF7F", "#7F00FF", "#C0C0C0", "#808080",
            "#400080", "#800040", "#804000", "#008040", "#408000", "#800080", "#408080", "#008080",
            "#804040", "#804080", "#408040", "#800000", "#008000", "#000080"
        ]
        n_colors = 0
        if type(label) == type(torch.Tensor()):
            n_colors = len(np.unique(label))
        elif type(label) == type(pd.DataFrame()):
            n_colors = len(np.unique(label.values))
        elif type(label) == type(pd.Series()):
            n_colors = len(np.unique(label.values.tolist()))
        elif type(label) == type(np.array([])):
            n_colors = len(np.unique(label.tolist()))


        if reducer is None:
            reducer = 'umap'
        z = self.get_latent_space(X)
        if self.latent_dim > 2 and reducer=='umap': 
            z = umap.UMAP(n_components=2).fit_transform(z)
        if self.latent_dim > 2 and reducer=='tsne': 
            z = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=35,random_state=2022).fit_transform(z)

        if require_distinguishable_colors is True:
            sns.scatterplot(x=z[:,0], y=z[:,1], hue=label, 
                    palette=sns.color_palette(custom_colors, n_colors)).set(title=title)
        else:
            sns.scatterplot(x=z[:,0], y=z[:,1], hue=label).set(title=title)
            
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    def get_cluster_embedding(self):
        r"""
        A helper function to get clusters in the embedding
        """
        return self.clusters.data

    def get_cluster_assignments(self, x):        
        r"""
        A helper function to get point assignments to clusters
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        z = self.ae.encoder(x)
        q = calc_q(z, self.clusters, self.alpha)
        return assign_cluster(q).detach().cpu().numpy()
    
    def get_cluster_abundance_matrix(self, X, sample_ids):
        r"""
        Get sample cluster abundance matrix.
        
        Args:
            X: gene expression. rows are cells, columns are genes
            sample_ids [list] or 1D[array]: sample id with same order as X 
            
        Returns:
            the sample cluster fraction matrix
        """
        info = pd.DataFrame({
                'sample_ids':sample_ids,
                'cluster':self.get_cluster_assignments(X),
            })
        cam = info.groupby(['sample_ids', 'cluster']).size().unstack()
        # cam = pd.DataFrame(0, index=temp.index, 
        #                         columns=list(range(self.n_clusters)))
        # cam.loc[:, temp.columns] = temp
        cam[np.isnan(cam)] = 0
        cam = cam.div(cam.sum(axis=1), axis=0)
        return cam

    def evaluate(self, X_train, y_train, epoch):
        # get sample cluster fraction matrix from traininig data
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        X_new_train = self.get_cluster_abundance_matrix(X_train, self.id_train)
        # get sample labels
        info_train = pd.DataFrame({
            'id':self.id_train,
            'label':y_train
        })
        y_true_train = info_train.groupby(['id', 'label']).size().index.to_frame()['label'].to_list()

        # get sample cluster fraction matrix from testing data
        X_new_test = self.get_cluster_abundance_matrix(self.X_test, self.id_test)
        # get sample labels 
        info_test = pd.DataFrame({
            'id':self.id_test,
            'label':self.y_test
        })
        y_true_test = info_test.groupby(['id', 'label']).size().index.to_frame()['label'].to_list()

        clf = LogisticRegression().fit(X_new_train, y_true_train)
        y_pred_train = clf.predict(X_new_train)
        y_pred_test = clf.predict(X_new_test)

        # training data
        self.accuracy_train.append(accuracy_score(y_true_train, y_pred_train))
        self.roc_auc_train.append(roc_auc_score(y_true_train, clf.predict_proba(X_new_train)[:, 1]))

        # testing data
        self.accuracy_test.append(accuracy_score(y_true_test, y_pred_test))
        self.roc_auc_test.append(roc_auc_score(y_true_test, clf.predict_proba(X_new_test)[:, 1]))

        # evaluation on current training epoch to add loss values
        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)
        self.ae.eval()
        with torch.no_grad():
            X_bar = self.ae(X_train)
            z = self.ae.encoder(X_train)
            q = calc_q(z, self.clusters, self.alpha)
            rec_loss = F.mse_loss(X_bar, X_train)
            ent_loss = calc_entropy(q, torch.tensor(y_train).to(self.device))
            kl_loss = F.kl_div(q.log(), self.P) # self.P is the target distribution from X_train
            self.loss_c.append(kl_loss.item())
            self.loss_r.append(rec_loss.item())
            self.loss_p.append(ent_loss.item())
            self.loss_total.append(kl_loss.item() + ent_loss.item() + rec_loss.item())
        # clean up space
        del X_bar, q, X_train, y_train, clf, X_new_train, X_new_test, info_train, info_test
        gc.collect()
        torch.cuda.empty_cache()

        # plot. might slow down program
        if self.plot_evaluation:
            epochs = np.arange(1, epoch+2)
            clear_output(wait=True)
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 13))
            from matplotlib.ticker import MaxNLocator
            axes[0,0].plot(epochs, self.loss_total, label='total loss')
            axes[0,0].legend(loc='lower left')
            axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0,1].plot(epochs, self.loss_p, label="ent loss")
            axes[0,1].legend(loc='lower left')
            axes[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0,2].plot(epochs, self.loss_c, label="cluster loss")
            axes[0,2].legend(loc='lower left')
            axes[0,2].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1,0].plot(epochs, self.loss_r, label="recstr loss")
            axes[1,0].legend(loc='lower left')
            axes[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1,1].plot(epochs, self.accuracy_train, color='blue', label='accuracy')
            axes[1,1].plot(epochs, self.accuracy_test, color='red', label='val accuracy')
            axes[1,1].legend(loc='lower left')
            axes[1,1].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1,2].plot(epochs, self.roc_auc_train, color='blue', label='roc-auc')
            axes[1,2].plot(epochs, self.roc_auc_test, color='red', label='val roc-auc')
            axes[1,2].legend(loc='lower left')
            axes[1,2].xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.show()
            if self.fold_num != None and epoch == self.epoch_train - 1:
                plt.draw() # NOTE
                fig.savefig('train_fold_{}.png'.format(self.fold_num))


# NOTES
# order preserving: 
#   self.ae.encoder,  TODO? assign_cluster(q)