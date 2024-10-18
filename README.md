# Phenotype Associated Single Cell encoder (PASCode)

Phenotype Associated Single Cell encoder (PASCode) is a deep learning framework for phenotype scoring of single cells. PASCode ensembles multiple Differential Abundance (DA) tools through a Robust Rank Aggregation (RRA) algorithm, and uses a graph attention network (GAT) to robustly and accurately annotate phenotype associated cell (PAC) scores for single cells. Given single-cell sequencing data and a contrastive pair of phenotypic labels (e.g., disease vs. control), PASCode outputs PAC scores for each cell. PASCode not only outperforms individual DA tools but also can predict PAC scores for individuals with unknown phenotype labels.

PASCode integrates existing DA tools, the RRA algorithm, and a trainable GAT model for PAC score annotation. PASCode offers a straighforward interface for easy access to these tools, all in the python environment. PASCode simplifies the usage of DA tools, RRA, and the GAT model by providing unified function calls, enabling DA analysis with standardized inputs and outputs.

![flowchart](./images/flowchart.png)

Users are advised to refer to our tutorials (*Tutorial_PASCode-RRA.ipynb*, *Tutorial_PASCode-ScorePrediction.ipynb*) for usage of PASCode, and understand how to best utilize PASCode for their own purposes (e.g., donor subsampling approach, DA tool options and parameters, GAT training, etc.).

## System requirements and dependencies
The code has been tested on Ubuntu 20.04 and Windows 12 with the following dependencies:

python==3.10.12

numpy==1.26.4\
scipy==1.14.1\
scanpy==1.10.2\
pandas==2.0.3\
anndata==0.10.3\
multianndata==0.0.4\
matplotlib==3.9.1\
seaborn==0.13.2\
cna==0.1.6\
meld==1.0.2\
rpy2==3.5.16\
torch==2.3.0\
torch_scatter=2.1.2\
torch_sparse=0.6.18\
torch_geometric==2.3.1\
scikit-learn==1.5.2

#### Note:
Current PASCode is built upon the following R package / DA methods versions:

Milo: milopy github repository as of Sep. 28 2024 \
MELD: v1.0.2 \
CNA: v0.1.6 \
DAseq: v1.0.0 \
RobustRankAggreg: v1.2.1 \
edgeR: v4.2.1

## Installation

PASCode is built upon existing DA tools and R packages, thus the user should install those tools (step 1) first before installing PASCode (step 2).

### Step 1: Install DA tools and RRA (2~3 min)

- *RobustRankAggreg* must be installed for Robust Rank Aggregation (RRA) to get aggregated cell labels. **(R)**

- *Milo* should be installed (follow instructions in https://github.com/emdann/milopy) if the user wants to include it in the RRA option. **(Python)**

- *edgeR* must be installed in order to run **Milo**. **(R)**

- *DAseq* should be installed (follow instructions in https://github.com/KlugerLab/DAseq) if the user wants to include it in the RRA option. **(R)**

### Step 2: Install PASCode (2~3 min)

After completing step 1:

1) Download the PASCode repository from github either by downloading the repository zip file or using git commands:
```python
git clone https://github.com/daifengwanglab/PASCode
```

2) Navigate to the PASCode directory, and run
```
pip install -r requirements.txt
```

#### Note:
The user may run into errors regarding *sparse tensor*. This is an existing issue (see https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-7970609) with the installation of PyG. In *requirements.txt*, we provided the wheel links to *torch_scatter* and *torch_sparse* to facilitate smooth installation, but that is for torch version 2.3.0 with CPU only.

To install *torch*, *torch-scatter*, *torch-sparse* for cuda, follow PyTorch installation guide and look for the corresponding wheel in https://data.pyg.org/whl/

For instance, the following commands install *torch-scatter* and *torch-sparse* with *torch-2.3.0*, *python version 3.10*, *cuda version 12.1* on a *linux* machine.
```
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu121/torch_sparse-0.6.18%2Bpt23cu121-cp310-cp310-linux_x86_64.whl
```

## Quick start

Note that this score function may not suit all scenarios due to its fixed settings over every step.

Users are advised to refer to our tutorials (**Tutorial_PASCode-RRA.ipynb**, **Tutorial_PASCode-ScorePrediction.ipynb**) for more controls over the process to best utilize PASCode for their own purposes (e.g., donor subsampling approach, DA tool options and parameters, GAT training, etc.).

```python
import PASCode
import scanpy as sc

adata = sc.read_h5ad('./data/synth_demo_2v6.h5ad')

# specify column names for subject ID and condition, and positive/negative conditions
subid_col = 'subject_id' 
cond_col = 'phenotype'
pos_cond = 'cond1'
neg_cond = 'cond2'

PASCode.model.score(
    adata=adata,
    subid_col=subid_col,
    cond_col=cond_col,
    pos_cond=pos_cond,
    neg_cond=neg_cond,
)

sc.pl.umap(adata, color=['PAC_score', 'phenotype', 'celltype'])
```

![demo_plot](./images/demo_plot.png)

Demo running time (including running Milo, MELD, DAseq, and traininig GAT model): 3min.

PASCode running time can vary across systems and various use cases, users are advised to follow our tutorials (*Tutorial_PASCode-RRA.ipynb*, *Tutorial_PASCode-ScorePrediction.ipynb*) to understand how to customize their PAC scoring process efficiently and accurately.
