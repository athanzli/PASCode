# Phenotype Associated Single Cell encoder (PASCode)

Phenotype Associated Single Cell encoder (PASCode) is a machine learning framework for phenotype scoring of single cells. PASCode uses a graph neural network to ensemble multiple Differential Abundance (DA) tools and robustly scores phenotype association of single cells. PASCode not only outperforms individual tools but also can transfer its latent representations to predict the PACs of individuals without known phenotypes. 

We used those PACs to prioritize novel cell subpopulations within and across different AD/NPS phenotypes for the PsychAD consortium.

![PASCodeworkflow](https://github.com/daifengwanglab/PASCode/assets/109684042/f20719a9-241e-4631-9cbb-448388fc1df2)

PASCode provides both pre-trained models and training from scratch for the annotation of Phenotype Associated Cell (PAC) scores:
* Pre-trained models: we provide models pre-trained on the PsychAD consortium for AD and NPS PAC score predictions.
* Training from scratch: the user can also use Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) by running the script we provided in our pipeline for cell aggregated labels, train the Graph Attention Network (GAT) on such labels and use the trained model for PAC score prediction.

## Dependencies
The script is based on python 3.10 above and requires the following packages:
- numpy >= 1.24.4
- pandas >= 1.5.3
- scipy >= 1.11.1
- scikit-learn >= 1.2.2
- scanpy >= 1.9.3
- anndata >= 0.10.2
- multianndata >= 0.0.4
- cna >= 0.1.4
- meld >= 1.0.0
- rpy2 >= 3.5.12
- anndata2ri >= 1.2
- pyreadr >= 0.4.7
- torch >= 2.0.0+cu118
- torch-geometric >= 2.3.1
- torch-sparse >= 0.6.17+pt20cu118
- torchvision >= 0.15.1+cu118

Some DA tools are either based on R programming languages or provided only by Github code, which can be installed according to:
- DAseq: https://github.com/KlugerLab/DAseq
- Milo: follow instructions on https://github.com/emdann/milopy/tree/master and put **milopy** under the **PASCode** directory.

## Download code
```python
git clone https://github.com/daifengwanglab/PASCode
```

## Usage

### Using pre-trained model

We provide pre-trained GAT models for AD, AD progression, Sleep Weight Gain Guilt Suicide, WeightLoss PMA and Depression Mood. Users are advised to follow our tutorial on input data preprocessing and the usage of such models in **PASCodePretrainedAnnotation.py** under the **tutorials** directory.

To load a pre-trained GAT model and predict PAC scores, simply follow:

```python
import PASCode
import torch

model = PASCode.model.GAT(
    in_channels=adata.X.shape[1], out_channels=64, num_class=3, heads=4)
model.load_state_dict(torch.load('./trained_model.pt'))

adata.obs['pac_score'] = model.predict(
    PASCode.Data().adata2gdata(adata)) # NOTE prediction
```

```python
sc.pl.umap(adata, color=['pac_score', 'syn_label', 'broad.cell.type'])
```
![output](https://github.com/daifengwanglab/PASCode/assets/109684042/9d96f755-27b9-4966-9ddd-74c3624719ea)

### Training models from scratch with DA tools
This involves four steps: 
1) input data preprocessing and graph construction. 
2) the user chooses which Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) to run for getting cell aggregated labels. 
3) train the Graph Attention Network (GAT) with such labels. 
4) use the trained model for PAC score prediction.

Users are advised to follow our tutorial on input data preprocessing and the usage of such models in **PASCodeFromScratch.py** under the **tutorials** directory.

## License
MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
