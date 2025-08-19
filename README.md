# Phenotype Associated Single Cell encoder (PASCode)

<!-- Badges: keep the set small and useful -->
[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101%2F2024.11.01.24316586-b31b1b)](https://www.medrxiv.org/content/10.1101/2024.11.01.24316586v1)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)](#system-requirements-and-dependencies)
[![OS](https://img.shields.io/badge/OS-Linux%20%7C%20Windows-informational)](#system-requirements-and-dependencies)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia)](#installation)
<!-- [![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE) -->

Phenotype Associated Single Cell encoder (PASCode) is a computational framework for phenotype scoring at the single-cell level. It integrates multiple differential abundance (DA) methods through an ensemble approach and leverages a graph attention network (GAT) to predict phenotype association scores for phenotype associated cells(PAC scores). Given single-cell sequencing data and a contrastive phenotypic label pair (e.g., disease vs. control), PASCode infers a PAC score for each cell, outperforming individual DA methods. PASCode combines existing DA methods, the Robust Rank Aggregation (RRA) algorithm, and trainable GAT models into unified Python-based interface. By standardizing inputs and outputs, it streamlines DA analysis by simplifying the running of these methods via user-friendly function calls.

![flowchart](./images/flowchart.png)

## System requirements and dependencies
The code has been tested on Ubuntu 20.04, 22.04 and Windows 12 with the following dependencies

Python version (**Note:** potential version issues may arise if using python>=3.13 or <=3.9. We recommend installing PASCode in a new environment (e.g., a conda environment `conda create -n PASCode python==3.10.12`) with the following tested python version)
```
python==3.10.12
```
Python packages
```
platformdirs>=2.5.0
numpy==1.26.4
scipy==1.14.1
scanpy==1.10.2
pandas==2.0.3
anndata==0.10.3
multianndata==0.0.4
matplotlib==3.9.1
seaborn==0.13.2
cna==0.1.6
meld==1.0.2
rpy2==3.5.16
torch==2.3.0
torch_scatter=2.1.2
torch_sparse=0.6.18
torch_geometric==2.3.1
scikit-learn==1.5.2
```
R packages / DA methods:
```
Milo: milopy github repository as of Sep. 28 2024 
MELD: v1.0.2 
CNA: v0.1.6 
DAseq: v1.0.0 
RobustRankAggreg: v1.2.1
edgeR: v4.2.1
```
## Installation

PASCode is built upon existing DA methods and R packages, thus the user should install some of those methods (step 1) first before installing PASCode (step 2).

### Step 1: Install DA methods and RRA (2~3 min)

- The `RobustRankAggreg` R package must be installed for Robust Rank Aggregation (RRA) to get single-cell aggregated phenotype labels **(R)**. 
    ```r
    install.packages('RobustRankAggreg')
    ```
- *Milo* (python version) has already been integrated in PASCode, thus only the following two dependencies need to be installed to run *Milo* **(R)**:

    ```r
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("edgeR")
    install.packages('statmod')
    ```

- `DAseq` should be installed to be included in RRA **(R)**.

  Follow instructions in https://github.com/KlugerLab/DAseq

- The rest of the DA methods will be automatically installed in step 2.

### Step 2: Install PASCode (2~3 min)

After completing step 1:

1) Download the PASCode repository from github, either by downloading the repository zip file or using git commands in a directory of interest:
```python
git init
git clone https://github.com/daifengwanglab/PASCode
```

2) Navigate to the PASCode directory, and run
```
pip install -r requirements.txt
```

3) install torch-scatter and torch-sparse (in this order)
- If on Linux Ubuntu:
```python
# first install torch-scatter
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-linux_x86_64.whl
# then install torch-sparse
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp310-cp310-linux_x86_64.whl
```
- If on Windows:
```python
# first install torch-scatter
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-win_amd64.whl
# then install torch-sparse
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp310-cp310-win_amd64.whl
```
Note:
- If this doesn't work, or an error regarding torch-sparse or torch-scatter is raised later when running GAT, you may consider checking the corresponding wheel version from https://data.pyg.org/whl/ and replace them here.
- If you run into errors regarding *sparse tensor*: This is a common issue. See https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-7970609 for the solution.

## Usage guide
If the user only wants a quick run to get PAC scores without caring much about customizations for better accuray and controls, use this function:

```python
import scanpy as sc
adata = sc.read_h5ad('./data/synth_demo_2v6_sparse.h5ad')
pac_scores = PASCode.model.score(
    adata=adata,
    subid_col='subject_id', # column name in adata.obs for subject IDs
    cond_col='phenotype', # column name for subject-level condition of interest
    pos_cond='cond1', # name for positive condition in `cond_col`
    neg_cond='cond2', # name for negative condition in `cond_col
    device='cuda:0' # GAT device. change accordingly
)
```

Below we present a general step-by-step usage guide.

More detailed tutorials are also provided. They can be found at `Tutorial_PASCode-RRA.ipynb` and `Tutorial_PASCode-ScorePrediction.ipynb` on this GitHub page.

### Step 1: Input data format
The input data should be an [anndata](https://anndata.readthedocs.io/en/stable/) object, with **already preprocessed** single-cell measurements data in `anndata.X` (`numpy.ndarray`) and subject-level information in `anndata.obs` (`pandas.DataFrame`). For example,

```python
import scanpy as sc
adata = sc.read_h5ad('./data/synth_demo_2v6_sparse.h5ad')
print(adata)
```
Example output
```
AnnData object with n_obs × n_vars = 11674 × 2000
    obs: 'celltype', 'phenotype', 'subject_id'
```
Take a look at `adata.X`
```python
adata.X = adata.X.toarray()
print(adata.X)
```
Example output
```
[[0.        0.        0.        ... 0.        1.1308908 0.       ]
 [0.        0.        0.        ... 0.        0.        0.       ]
 [0.        0.        0.        ... 0.        0.        0.       ]
 ...
 [0.        0.        0.        ... 0.        0.        0.       ]
 [2.4369774 0.        0.        ... 2.4369774 0.        0.       ]
 [0.        0.        0.        ... 0.        0.        0.       ]]
```
Take a look at `adata.obs`
```python
import anndata
print(adata.obs)
```
Example output
```
                    celltype phenotype subject_id
TAG                                              
AACTGGTGTACCGGCT.1        Ex     cond1         21
AAGACCTAGTTAACGA.1        Ex     cond1         21
ACATCAGCAGGCTCAC.1        Ex     cond1          2
ACGAGGATCCATTCTA.1        Ex     cond1          2
ATCATCTGTGGACGAT.1        Ex     cond2         34
...                      ...       ...        ...
GCACATACATGCCTTC.40      End     cond2         43
GCCTCTACACTTAACG.44      Per     cond1         21
GTGCGGTCAATCGGTT.46      End     cond2         29
CACACTCTCTCTGAGA.48      Per     cond1          2
TGCCCATAGTAGGTGC.48      Per     cond1         21

[11674 rows x 3 columns]
```
Make sure `adata.obs` (`pandas.DataFrame`) has at least the following:
- single-cell IDs as indices (e.g. the 'TAG' indices 'AACTGGTGTACCGGCT.1, ...')
- a *subject ID* column for the IDs of subjects (e.g., 'subject_id' above).
- a *condition* column, indicating either a positive condition (e.g., AD) or a negative condition (e.g., Control) for the subject (e.g., 'phenotype' above).

We can take a look at subject-level information w.r.t. any subject-level labels of interest, together with subject ID:

```python
subid_col = 'subject_id' # specify the column name for subject IDs
import PASCode
dinfo = PASCode.utils.subject_info(
    adata.obs,
    subid_col=subid_col,
    # specify the subject-level column names to summarize (must be subject-level)
    columns=['Sex', 'Age'] # e.g. count subject numbers for sex and age
)
print(dinfo)
```
Example output
```
       Sex Age  cell_num
43    Male  81      1415
24    Male  86      1425
34    Male  80      1438
29  Female  73      1454
31  Female  73      1463
2   Female  83      1484
21    Male  62      1489
36  Female  62      1506
```

If you need to **create a new** `anndata` object, you can easily do so via the `andata.AnnData` class by providing at least
- a preprocessed single-cell measurements `X` (e.g., single-cell RNA-seq) with cells as rows and features as columns. \
**Note: make sure `X` is preprocessed, at the very least log-normalized**, if not standard-scaled (PASCode will automatically perform standard scaling, which is strongly recommended for procedures like PCA and GAT training to achieve higher accuracy and stability).
- a single-cell observation information dataframe, including celltype, phenotype, subject_id, etc.
```python
adata = anndata.AnnData(X=X, obs=obs)
```

### Step 2: Balance donor numbers via subsampling

Skip this step if subject numbers in the contrastive condition pair are already balanced.

In our benchmark experiments, we found that DA methods are more accurate for balanced donor numbers between the positive condition ($n_1$) and negative condtion ($n_2$). Unless $n_1$ and $n_2$ are extremely close, subsampling is almost always recommended to alleviate performance degradation.

Subject subsampling is particularly necessary for imbalanced population-scale datasets. For example, for PsychAD's AD contrast, we have 314 AD subjects and 111 Controls, which is highly imbalanced. Running DA and RRA directly would lead to low accuracy, inhibiting meaningful downstream analysis.

Therefore, in our analysis, we took 100 AD vs. 100 Controls as the train set for running DA methods and RRA to get *aggregated phenotype labels*, then trained a GAT model with this train set and a 11 AD vs. 11 Controls validation set (for early stopping to obtain the best model). We then used the trained GAT model to infer PAC scores for the whole dataset (i.e., 314 AD vs. 111 Controls).
 
```python
"""This function will automatically subsample subjects to the smaller number in the two conditions.
For instance, 2v6 -> 2v2
"""
# adata0 will be used for complete PAC score annotation using GAT later in Step 4
adata0 = adata.copy()

# specify required column names, and the labels for the contrastive condition pair
subid_col = 'subject_id'
cond_col = 'phenotype'
print(adata.obs[cond_col].value_counts())
pos_cond = 'cond1'
neg_cond = 'cond2'

import PASCode
adata = PASCode.utils.subsample_donors(
    adata=adata,
    subid_col=subid_col,
    cond_col=cond_col,
    pos_cond=pos_cond,
    neg_cond=neg_cond,
    sex_col=None, # Specify the column name for sex in adata.obs to account for sex balance during subsampling. Default: None.
)
```
Example output
```
Before donor subsampling:
phenotype
cond2    6
cond1    2
Name: count, dtype: int64
'subsample_num' not provided. Automatically subsampling to the smaller number of subjects in the two conditions.
Donor subsampling:  2:2
After donor subsampling:
phenotype
cond2    2
cond1    2
Name: count, dtype: int64
```
### Step 3: Run DA methods and RRA
We have a convenient function call to run DA methods and RRA:
```python
import PASCode
adata.obs['aggreg_label'] = PASCode.da.agglabel(
    adata,
    subid_col,
    cond_col,
    pos_cond,
    neg_cond,
    da_methods=['milo','meld','daseq'] # recommended/default combination that yields high accuracy
)
```
Example output
```
============================= DA and RRA... =============================
'use_rep' not found in adata.obsm. Computing PCA from adata.X to use as rep for cell-cell graph...
Scaling...
Running PCA...
Computing connectivities...

----------------------------- Milo started ... -----------------------------
Making neighborhoods...
WARNING:root:Using X_pca as default embedding
Counting neighborhoods...
Running differential abundance testing...
----------------------------- Milo Time cost (s):  2.58  -----------------------------


 ----------------------------- MELD started ... -----------------------------
Building graph on 5942 samples and 50 features.
Calculating graph and diffusion operator...
  Calculating KNN search...
  Calculated KNN search in 2.00 seconds.
  Calculating affinities...
  Calculated affinities in 0.13 seconds.
Calculated graph and diffusion operator in 2.41 seconds.
----------------------------- MELD Time cost (s):  2.69  -----------------------------


    WARNING: The R package "reticulate" only fixed recently
    an issue that caused a segfault when used with rpy2:
    https://github.com/rstudio/reticulate/pull/1188
    Make sure that you use a version of that package that includes
    the fix.
    
----------------------------- DAseq started ... -----------------------------
Calculating DA score vector.
Running GLM.
Test on random labels.
Setting thresholds based on permutation
----------------------------- DA-seq Time cost (s):  47.72  -----------------------------



----------------------------- RobustRankAggregation started ... -----------------------------
Aggregating positive score ranks...
Aggregating negative score ranks...
Combining positive and negative score ranks...
----------------------------- RobustRankAggregation Time cost (s): 0.37 -----------------------------


============================= DA and RRA Time cost (s):  63.49  =============================

aggreg_label
 0.0    4190
 1.0     887
-1.0     865
Name: count, dtype: int64
```

### Step 4: GAT for annotating PAC scores for the whole dataset

We first need to transfer the aggregated labels from the subsampled dataset back to the original dataset.
```python
adata_sub = adata.copy() # subsampled balanced dataset
adata = adata0.copy() # original whole dataset
adata.obs.loc[adata_sub.obs.index, 'aggreg_label'] = adata_sub.obs['aggreg_label']
```
Prepare training and validation masks
```python
PASCode.model.get_val_mask(
    adata_sub, 
    subid_col=subid_col,
    cond_col=cond_col,
    pos_cond=pos_cond,
    neg_cond=neg_cond
)

# Assign masks to the original dataset
adata.obs['val_mask'] = adata.obs.index.isin(
    adata_sub.obs[adata_sub.obs['val_mask']].index
)
adata.obs['train_mask'] = adata.obs.index.isin(
    adata_sub.obs[adata_sub.obs['train_mask']].index
)
```

Train the GAT model
```python
model = PASCode.model.train_model(
    adata=adata,
    agglabel_col='aggreg_label',
    device='cuda:0' # NOTE change accordingly
)
```
Example output
```
No graph found in `adata.obsp`'s `connectivities`. Building graph from scratch...
Scaling data...
Running PCA...
Builidng graph...
Building graph time cost (s): 110.534112.
Constructing batches...
Computing METIS partitioning...
Done!
Batch construction done.

============================= Training GAT... =============================
Epoch: 1/100 - lr: 1.000e-03 - train_loss: 0.449 - val_loss: 0.392 
Epoch: 2/100 - lr: 1.000e-03 - train_loss: 0.194 - val_loss: 0.342 
Epoch: 3/100 - lr: 1.000e-03 - train_loss: 0.163 - val_loss: 0.193 
Epoch: 4/100 - lr: 1.000e-03 - train_loss: 0.140 - val_loss: 0.208 
Epoch: 5/100 - lr: 1.000e-03 - train_loss: 0.128 - val_loss: 0.172 
Epoch: 6/100 - lr: 1.000e-03 - train_loss: 0.110 - val_loss: 0.193 
Epoch: 7/100 - lr: 1.000e-03 - train_loss: 0.097 - val_loss: 0.196 
Epoch: 8/100 - lr: 1.000e-03 - train_loss: 0.091 - val_loss: 0.174 
Epoch: 9/100 - lr: 5.000e-04 - train_loss: 0.092 - val_loss: 0.172 
Epoch: 10/100 - lr: 5.000e-04 - train_loss: 0.085 - val_loss: 0.184 
Epoch: 11/100 - lr: 5.000e-04 - train_loss: 0.075 - val_loss: 0.168 
Epoch: 12/100 - lr: 5.000e-04 - train_loss: 0.076 - val_loss: 0.185 
Epoch: 13/100 - lr: 5.000e-04 - train_loss: 0.078 - val_loss: 0.180 
Epoch: 14/100 - lr: 5.000e-04 - train_loss: 0.074 - val_loss: 0.175 
Epoch: 15/100 - lr: 2.500e-04 - train_loss: 0.069 - val_loss: 0.172 
Epoch: 16/100 - lr: 2.500e-04 - train_loss: 0.072 - val_loss: 0.171 
Epoch: 17/100 - lr: 2.500e-04 - train_loss: 0.070 - val_loss: 0.175 
Epoch: 18/100 - lr: 1.250e-04 - train_loss: 0.066 - val_loss: 0.178 
Epoch: 19/100 - lr: 1.250e-04 - train_loss: 0.061 - val_loss: 0.171 
Epoch: 20/100 - lr: 1.250e-04 - train_loss: 0.062 - val_loss: 0.175 
Epoch: 21/100 - lr: 6.250e-05 - train_loss: 0.061 - val_loss: 0.172 
Epoch: 22/100 - lr: 6.250e-05 - train_loss: 0.066 - val_loss: 0.172 
============================= Training Time cost (s): 19.68 =============================
```

Predict PAC scores and visualize
```
adata.obs['PAC_score'] = model.predict(adata)

sc.tl.umap(adata)
sc.pl.umap(adata, color=['PAC_score', 'phenotype', 'celltype'])
```

![alt text](./images/demo_plot.png)

Demo total running time (including running Milo, MELD, DAseq, and traininig GAT model): ~3min.

PASCode running time can vary across systems and data scales.

## Reference

Chenfeng He*, Athan Z. Li*, Kalpana Hanthanan Arachchilage*, Chirag Gupta*, Xiang Huang, Xinyu Zhao, PsychAD Consortium, Kiran Girdhar, Georgios Voloudakis, Gabriel E. Hoffman, Jaroslav Bendl, John F. Fullard, Donghoon Lee, Panos Roussos†, Daifeng Wang†. *Phenotype Scoring of Population Scale Single-Cell Data Dissects Alzheimer's Disease Complexity*. doi: https://doi.org/10.1101/2024.11.01.24316586.