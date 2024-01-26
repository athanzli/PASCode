# Phenotype Associated Single Cell encoder (PASCode)

The complexity of Alzheimer’s disease (AD) progression manifests in diverse clinical phenotypes, including resilience and neuropsychiatric symptoms (NPSs). These variations primarily arise from elusive alterations in gene expression and regulation within specific brain cells. The PsychAD project has generated snRNA-seq data of 6 million cells across 1,495 individual brains, covering various AD phenotypes and NPSs. However, existing cell type-based analyses inadequately elucidate cell-phenotype relationships and underlying molecular mechanisms. To address this challenge, leveraging the single-cell gene expression data in PsychAD, our integrated machine learning analysis identified phenotype associated cells (PACs) for AD progression, resilience, and depression. Particularly, our approach, Phenotype Associated Single Cell encoder (PASCode), uses a graph neural network to ensemble multiple tools and robustly scores phenotype association of single cells. PASCode not only outperforms individual tools but also can transfer its latent representations to predict the PACs of individuals without known phenotypes. We used those PACs to prioritize novel cell subpopulations within and across different phenotypes. Subsequent gene analyses, including differentially expressed genes, gene signatures and regulatory modules provide further mechanistic insights for AD. For instance, we identified the excitatory neurons associated with AD progression and observed specific gene expression dynamics of AD resilience in the progression using single-cell trajectory analysis. Also, we identified astrocyte subpopulations associated with depression in AD, along with underlying gene networks and pathways such as inflammation and endoplasmic reticulum stress. Finally, we summarized a phenotypic single-cell atlas with open-source tools allowing exploration of additional cell subpopulations for diseases and phenotypes in PsychAD. 

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
- Milo: follow instructions on https://github.com/emdann/milopy/tree/master and put 'milopy' under the 'PASCode' directory.

## Download code
```python
git clone https://github.com/daifengwanglab/PASCode
```

## Usage

### Using pre-trained model

We provide pre-trained GAT models for AD, AD progression, Sleep Weight Gain Guilt Suicide, WeightLoss PMA and Depression Mood. Users are advised to follow our tutorial on input data preprocessing and the usage of such models in './demo/PASCodePretrainedAnnotation.py'

### Training models from scratch with DA tools
This involves four steps: 
1) input data preprocessing and graph construction. 
2) the user chooses which Differential Abundance (DA) tools and Robust Rank Aggregation (RRA) to run for getting cell aggregated labels. 
3) train the Graph Attention Network (GAT) with such labels. 
4) use the trained model for PAC score prediction.

Users are advised to follow our tutorial on input data preprocessing and the usage of such models in './demo/PASCodeFromScratch.py'

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
