# Phenotype Associated Single Cell encoder (PASCode)

Phenotype Associated Single Cell encoder (PASCode) is a machine learning framework for phenotype scoring of single cells. PASCode ensembles multiple Differential Abundance (DA) tools through a Robust Rank Aggregation (RRA) algorithm, and uses a graph neural network to robustly annotate the phenotype association scores for single cells. PASCode not only outperforms individual DA tools but also can transfer its latent representations to predict the PACs of individuals with unknown phenotypes. PASCode is built based on the python package _anndata_, taking an anndata object with graph and gene expression data as input, and produces PAC scores for each cell as output. PASCode is widely applicable to any pair of phenotypic labels.

![PASCodeworkflow](https://github.com/daifengwanglab/PASCode/assets/109684042/f20719a9-241e-4631-9cbb-448388fc1df2)

PASCode provides both training from scratch and pre-trained models for the annotation of Phenotype Associated Cell (PAC) scores:
* PASCode custom-training: the user can use Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) using our provided PASCode functions to get aggregated cell labels, and then train the Graph Attention Network (GAT) for PAC score annotations. This includes the following steps:

1) Step 1: Graph construction for the whole data.

2) Step 2: (For a donor-number-balanced subset) Graph construction. Run Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) to get aggregated cell labels for a donor-number-balanced subset.

3) Step 3: Train the Graph Attention Network (GAT) on the balanced subset. 

4) Step 4: Use the trained model for PAC score annnotation for the whole dataset.

* PASCode Pre-trained: we provide models pre-trained on the PsychAD consortium for direct AD and NPS PAC score predictions.

## Installation

### step 1: install R packages

- **RobustRankAggreg** need to be installed for Robust Rank Aggregation (RRA) to get aggregated cell labels.

- **DAseq** (follow instructions in https://github.com/KlugerLab/DAseq) need to be installed if the user wants to include it in the RRA option (it is included in the tutorial code).

- **edgeR** need to be installed in order to run **Milo** if the user wants to include it in the RRA option (it is included in the tutorial code).

Installation time: 2-5 minutes.

### step 2: install PASCode

After installing the R packages:

1) Download the PASCode code from github either by downloading the repository zip file directly or using git commands:
```python
git clone https://github.com/daifengwanglab/PASCode
```

2) run
```
pip install .
```

Installation time: 2-5 minutes.

## Usage

### Training models from scratch
This involves four steps (running time estimated using the 35k cells in the demo data, may depend on user choices and system):
1) Step 1: Graph construction for the whole data. Running time ~5min
2) Step 2: (For a donor-number-balanced subset) Graph construction. Run Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) to get aggregated cell labels for a donor-number-balanced subset. Running time ~10min
3) Step 3: Train the Graph Attention Network (GAT) on the balanced subset. Running time ~8 min (using an RTX A6000 Graphics card)
4) Step 4: Use the trained model for PAC score annnotation for the whole dataset. Running time <1min

Here we provide an example of the complete procedure.

First of all, we need to import the corresponding library.
```python
import PASCode
import scanpy as sc
import numpy as np
import torch

DATA_PATH = '../data/' # NOTE
```

In step 1, we need to load the anndata object, which is required to have preprocessed gene expression data stored in anndata.X, and information regarding conditions and donor IDs stored in anndata.obs:
```python
###############################################################################
# Step 1: build graph
###############################################################################
file_path = DATA_PATH + 'synth_demo.h5ad'
adata = sc.read_h5ad(file_path)

cond_col = 'syn_label' # condition column
pos_cond = 'cond1' # positive condition
neg_cond = 'cond2'  # negative condition
donor_col = 'subid' # donor id column

# assuming the expression data is already preprocessed and stored in adata.X
# if not, please preprocess the data first
# PASCode graph building is based on the "X_pca" field in adata.obsm
# if not, the function will automatically run sc.pp.pca to get the pca coordinates based on adata.X
PASCode.graph.build_graph(adata)
```

Step 2 subsamples donors to obtain a donor-number-balanced subset, which is then input to PASCode ensemble DA tools and RRA to get aggregated cell labels. Aggregated labels are assigned back to the original anndata object in the end.
```python
###############################################################################
# Step 2: subsample, build graph, and get aggregated labels
###############################################################################
adata_pac = PASCode.subsample.subsample_donors(
    adata=adata,
    subsample_num="6:6", # NOTE change according to the specific donor numbers in the dataset
    donor_col=donor_col,
    cond_col=cond_col,
    pos_cond=pos_cond,
    neg_cond=neg_cond,
    mode='random',
)

# build graph for the subsampled anndata object
PASCode.graph.build_graph(adata_pac)

# run DA tools and RRA to get cell aggregate dlabels
PASCode.da.agglabel(
    adata_pac,
    donor_col,
    cond_col,
    pos_cond,
    neg_cond,
    methods=['milo','meld','daseq']
)

# assign aggregated labels to the whole anndata object which has the whole graph
adata.obs.loc[adata_pac.obs.index, 'rra_pac'] = adata_pac.obs['rra_pac'].values
```

Step 3 trains a GAT model using the preprocessed anndata object from step 2.
```python
###############################################################################
# Step 3: train GAT model
###############################################################################
agglabel_col = 'rra_pac'
# this will train a GAT model and save the trained model to the current directory
# the user is advised to look for more details in the tutorial if they want to 
# customize certain steps in the training procedure
model = PASCode.model.train_gat(
    adata,
    agglabel_col,
    donor_col,
    cond_col
)
```

For step 4, PAC scores are annotated for all single cells in the original dataset using the trained GAT model.
```python
###############################################################################
# Step 4: using the trained model for PAC score predictions
###############################################################################
model.load_state_dict(torch.load('./trained_model.pt'))
adata.obs['pac_score'] = model.predict(
    PASCode.Data().adata2gdata(adata)
)
```

```python
###############################################################################
# save
###############################################################################
adata.write_h5ad("./anndata_with_pac_scores.h5ad")
```

For more details, users are advised to follow our tutorial on input data preprocessing and the usage of such models in **PASCodeFromScratch.py** under the **tutorials** directory.

### Using pre-trained model for direct PAC score annotations

We provide pre-trained GAT models for AD, AD progression, Sleep Weight Gain Guilt Suicide, WeightLoss PMA and Depression Mood. Users are advised to follow our tutorial on input data preprocessing and the usage of such models in **PASCodePretrainedAnnotation.py** under the **tutorials** directory. 

Running time < 1min

To load a pre-trained GAT model and predict PAC scores, simply follow:

```python
import PASCode
import torch

adata = sc.read_h5ad(DATA_PATH + "seaad.h5ad")

model = PASCode.model.GAT(
    in_channels=adata.X.shape[1], out_channels=64, num_class=3, heads=4)

# choose from c02_model, r01_model, c90_model, r91_model, c92_model
model.load_state_dict(torch.load('./pretrained_models/c02_model.pt'))

# predict
adata.obs['pac_score'] = model.predict(PASCode.Data().adata2gdata(adata))
```

```python
sc.pl.umap(adata, color=['pac_score', 'Subclass'])
```
![output](https://github.com/daifengwanglab/PASCode/assets/109684042/d62dbfcc-920a-4774-8b4a-c03a253899f8)

