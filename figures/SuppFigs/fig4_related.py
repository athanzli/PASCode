# %%
%reload_ext autoreload
%autoreload 2

import os, sys
sys.path.append('../..')
import PASCode
import gc
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PASCode.utils import plot_pac_umap, plot_umap, plot_legend

rand_seed = 0
PASCode.random_seed.set_seed(rand_seed)

import seaborn as sns
import scanpy as sc

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/PsychAD/'

#%% [markdown]
################################ preprocessing
suball = pd.read_csv("/home/che82/data/psychAD/metadata.csv", index_col=0)
adata_ad = sc.read_h5ad(DATA_PATH + 'c02_only_obs_obsm.h5ad')

###### NOTE TODO
nps = 'c92'
nps_cond_col = 'c92x'
subid_col = 'SubID'
ad_cond_col = 'c02x'
common_pos_pac_col = 'c02_c92_common_PAC+'
nps_pac_col = 'c92_PAC'

# nps = 'c90'
# nps_cond_col = 'c90x'
# subid_col = 'SubID'
# ad_cond_col = 'c02x'
# common_pos_pac_col = 'PAC_c02+_c90+'
# nps_pac_col = 'c90_PAC'

# nps = 'c91'
# nps_cond_col = 'c91x'
# subid_col = 'SubID'
# ad_cond_col = 'c02x'
# common_pos_pac_col = 'PAC_c02+_c91+'
# nps_pac_col = 'c91_PAC'

############
adata_nps = sc.read_h5ad(DATA_PATH + nps + '_only_obs_obsm.h5ad')

subinfo_ad = PASCode.utils.subject_info(
    adata_ad.obs,
    subid_col=subid_col,
    columns=['Sex', ad_cond_col])
subinfo_nps = PASCode.utils.subject_info(
    adata_nps.obs,
    subid_col=subid_col,
    columns=['Sex', nps_cond_col])
common_subj = np.intersect1d(subinfo_ad.index, subinfo_nps.index)
adata = adata_ad[adata_ad.obs[subid_col].isin(common_subj)]
adata.obs[nps_cond_col] = subinfo_nps.loc[adata.obs[subid_col], nps_cond_col].values
adata.obs[nps_cond_col[:-1]+'_pac_score'] = adata_nps.obs['pac_score'].loc[adata.obs.index]
adata.obs[ad_cond_col[:-1] + '_pac_score'] = adata.obs['pac_score'].values
subinfo = PASCode.utils.subject_info(
    adata.obs,
    subid_col=subid_col,
    columns=['Sex', ad_cond_col, nps_cond_col])
print(subinfo[ad_cond_col].value_counts())
print(subinfo[nps_cond_col].value_counts())

# add gxp
adata_gxp = sc.read_h5ad('/media/che82/hechenfon/pead_freeze25/datasets2.5_M_selgenes2.h5ad')
ind = [adata_gxp.obs.index.get_loc(i) for i in adata.obs.index]
import anndata
adata = anndata.AnnData(X=adata_gxp.X[ind, :], 
                        obs=adata.obs, 
                        var=adata_gxp.var, 
                        uns=adata.uns, 
                        obsm=adata.obsm, 
                        obsp=adata.obsp)
# 
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=5) # TODO tune this param, rerun.
# graph & leiden
adata.obs[nps + '_PAC'] = PASCode.pac.assign_pac(
    adata.obs[nps + '_pac_score'].values
)
adata.obs['c02_PAC'] = PASCode.pac.assign_pac(
    adata.obs['c02_pac_score'].values
)
adata.write_h5ad(DATA_PATH + 'c02_' + nps + '_overlap.h5ad')

#%% ########################## analyse
# adata = sc.read_h5ad(DATA_PATH + 'c02_' + nps + '_overlap.h5ad')
adata = sc.read_h5ad(DATA_PATH + 'c02_' + nps + '_overlap_only_obs_obsm.h5ad')

#%% try using discrete (cutoff 0.5)
adata.obs[common_pos_pac_col] = 0
mask = (adata.obs[nps_pac_col] == 1) & (adata.obs['c02_PAC'] == 1)
adata.obs.loc[mask, common_pos_pac_col] = 1
print(adata.obs[common_pos_pac_col].value_counts())

adata.obs['c02_PAC+'] = 0
mask = adata.obs['c02_PAC'] == 1
adata.obs.loc[mask, 'c02_PAC+'] = 1
print(adata.obs['c02_PAC+'].value_counts())

adata.obs[nps + '_PAC+'] = 0
mask = adata.obs[nps + '_PAC'] == 1
adata.obs.loc[mask, nps + '_PAC+'] = 1
print(adata.obs[nps + '_PAC+'].value_counts())


#%% for test only 
import pickle as pkl
with open('/media/che82/hechenfon/pead_freeze25/datasets2.5_M_protein_coding_info.pkl', 'rb') as f:
    gxp = pkl.load(f)
gxp

#%% ######################################## C02 PAC+ and NPS  PAC+
cat_col = 'leiden'
cats = adata.obs[cat_col].unique()

from matplotlib_venn import venn2
pvalues = {}
for cat in cats:
    # skip those cats that have none or too few AD-PAC+ or NPS-PAC+ or common PAC+
    thres = 100
    tb = adata.obs[adata.obs[cat_col]==cat].groupby(['c02_PAC+', nps + '_PAC+']).size().unstack()
    if ((1 not in tb.index) or (1 not in tb.columns)) or \
        (tb.loc[1].sum() < thres or tb[1].sum(0) < thres) or \
        (tb.loc[1, 1] < thres):
        continue

    # hypergeometric test
    M = adata.obs[adata.obs[cat_col]==cat].shape[0]
    n = (adata.obs[adata.obs[cat_col]==cat]['c02_PAC']==1).sum()
    N = (adata.obs[adata.obs[cat_col]==cat][nps_pac_col]==1).sum()
    x = (adata.obs[adata.obs[cat_col]==cat][common_pos_pac_col]==1).sum()
    rv = scipy.stats.hypergeom(M, n, N)
    pvalues[cat] = 1-rv.cdf(x)

    # just by frac NOTE already did this, bottom line: there are two subclusters 23, 8 identifeid as sigfniciant (proportion > 0.5) in addition to hypergeometric resutls. Both are in Oligo_OPALIN  & Oligo_RBFOX1. And they 
    # pvalues[cat] = tb.loc[1,1] / min(tb.loc[1].sum(), tb[1].sum(0))

print(pd.Series(pvalues).sort_values().values)
# remove insignificant cats (p>0.05)
pvalues = {k:v for k,v in pvalues.items() if v < 0.05}

pvs = list(pvalues.values())
pvs_ordered = np.array(pvs)[np.argsort(pvs)] + 1e-100
cats_ordered = np.array(list(pvalues.keys()))[np.argsort(pvs)]
custom_order = [cat for cat in cats_ordered]
sns.barplot(x = - np.log10(pvs_ordered), 
            y = cats_ordered, 
            # figure=plt.figure(figsize=(5, 14)),
            order=custom_order,
            color='purple'); 
plt.xlabel('-log10(p-value)')
plt.show()

#%% ########################### look at specific subcluster
sig_celltype = []
for cat in cats_ordered:
    adata.obs[adata.obs[cat_col]==cat].groupby(['c02_PAC+', nps + '_PAC+']).size().unstack()
    sr = adata.obs.groupby(['subtype', 'leiden']).size().unstack()[cat].sort_values(ascending=False)
    sig_celltype.append(sr.index[0])
print(sig_celltype)
#%% venn diagram
# adata.obs['c02_PAC+'] = 0
# adata.obs.loc[adata.obs['c02_PAC']==1, 'c02_PAC+'] = 1
# adata.obs[nps + '_PAC+'] = 0
# adata.obs.loc[adata.obs[nps_pac_col]==1, nps + '_PAC+'] = 1
# adata.obs[nps + '_PAC+'] = 0
# adata.obs.loc[adata.obs[nps_pac_col]==1, nps + '_PAC+'] = 1
cat = '16' # TODO
try:
    ad_pac_pos = adata.obs.loc[adata.obs[cat_col] == cat, 'c02_PAC'].value_counts()[1]
except KeyError:
    ad_pac_pos = 0
try:
    nps_pac_pos = adata.obs.loc[adata.obs[cat_col] == cat, nps_pac_col].value_counts()[1]
except KeyError:
    nps_pac_pos = 0
try:
    ad_nps_pac_pos = adata.obs.loc[adata.obs[cat_col] == cat, common_pos_pac_col].value_counts()[1]
except KeyError:
    ad_nps_pac_pos = 0
venn2(subsets=(ad_pac_pos - ad_nps_pac_pos, nps_pac_pos - ad_nps_pac_pos, ad_nps_pac_pos),
        set_colors=('red', 'orange'))
plt.title(cat)
plt.legend(('AD PAC+', 'NPS PAC+'))
plt.show()


#%% ############################### umap
## Visualization of cell types and PAC scores
### subclass
#%%
adata = sc.read_h5ad(DATA_PATH + 'PsychAD/c02_' + nps + '_overlap.h5ad')
# adata = sc.read_h5ad(DATA_PATH + 'c02_' + nps + '_overlap_only_obs_obsm.h5ad')
# adata = sc.read_h5ad(DATA_PATH + 'rosmap/c02_rosmap.h5ad')
psychad_palette = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/figures/PsychAD_color_palette_230921.csv', index_col=0)

class_palette = psychad_palette[psychad_palette['name'].isin(adata.obs['class']) & (psychad_palette.index == 'class')]
class_palette = class_palette.sort_values(by='name')
class_palette = dict(zip(
    class_palette['name'].values,
    class_palette['color_hex'].values))
subclass_palette = psychad_palette[psychad_palette['name'].isin(adata.obs['subclass']) & (psychad_palette.index == 'subclass')]
subclass_palette = subclass_palette.sort_values(by='name')
subclass_palette = dict(zip(
    subclass_palette['name'].values, 
    subclass_palette['color_hex'].values))

c02 = sc.read_h5ad(DATA_PATH + 'c02_only_obs_obsm.h5ad')
ind = [c02.obs.index.get_loc(i) for i in adata.obs.index]
adata.obsm['c02_X_umap'] = c02.obsm['X_umap'][ind]
adata.write_h5ad(DATA_PATH + 'c02_' + nps + '_overlap.h5ad')

#%%
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='c02_X_umap',
    class_col='subclass', 
    class_palette=subclass_palette,
    text_on_plot=True,
    save_path='./c02_' + nps + '_ovlp_subclass_using_c02_umap.tiff'
)
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='c02_X_umap',
    class_col='subclass', 
    class_palette=subclass_palette,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_subclass_using_c02_umap_no_text.tiff'
)
plot_legend(
    legend_handles=handles,
    legend_labels= labels, #[l.split(':')[0] for l in labels],
    legend_ncol=3,
    save_path='./c02_' + nps + '_ovlp_subclass_legend.pdf'
)

handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='c02_X_umap',
    class_col='leiden', 
    class_palette=None,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_leiden_umap_2.tiff',
    use_default_colors=True
)
plot_legend(
    legend_handles=handles,
    legend_labels= [l.split(':')[0] for l in labels], #labels, #[l.split(':')[0] for l in labels],
    legend_ncol=3,
    save_path='./c02_' + nps + '_ovlp_subclass_legend_2.pdf'
)

sc.pl.umap(
    adata,
    color='leiden',
    umap_key='c02_X_umap'
)


#%%
adata.obs['leiden_sig_cluster_c02+_' + nps + '+'] = 'insignificant'
for sig_cluster in list(pvalues.keys()):
    adata.obs.loc[adata.obs['leiden']==str(sig_cluster), 'leiden_sig_cluster_c02+_' + nps + '+'] = str(sig_cluster)
adata.obs['leiden_sig_cluster_c02+_' + nps + '+'] = adata.obs['leiden_sig_cluster_c02+_' + nps + '+'].astype('category')

# # print(list(pvalues.keys()))
# sig_leiden_palette = {
#     'insignificant': 'grey',
#     '24': 'purple',
#     '22': '#0000FF',
#     '16': 'green',
#     '8': '#FFFF00',
#     '48': '#FF0000',
#     '45': '#42eff5'
# }

sig_leiden_palette = {
    'insignificant': 'grey',
    '19': 'purple',
    '29': '#0000FF',
    '4': 'green',
    '44': '#FFFF00',
    '6': '#FF0000',
    # '45': '#42eff5'
}


#%%
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='c02_X_umap',
    class_col='leiden_sig_cluster_c02+_' + nps + '+', 
    class_palette=sig_leiden_palette,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_sig_leiden_c02+_' + nps + '+_using_c02_umap.tiff'
)

plot_legend(
    legend_handles=handles,
    legend_labels= [l.split(':')[1] for l in labels],
    legend_ncol=1,
    save_path='./c02_' + nps + '_ovlp_sig_leiden_c02+_' + nps + '+_legend.pdf'
)

#%% [markdown] 
## PAC  umap
# plot_pac_umap(
#     adata, 
#     umap_key='c02_X_umap', 
#     pac_col='c02_PAC_score',
#     save_path='./c02_' + nps + '_ovlp_c02_PAC_using_c02_umap.tiff'
# )

plot_pac_umap(
    adata, 
    umap_key='c02_X_umap', 
    pac_col='' + nps + '_PAC_score',
    save_path='./c02_' + nps + '_ovlp_tscore_' + nps + '_PAC_using_c02_umap.tiff',
    colors=['#9e471b', '#ffffff', '#0e38c2'] # blue, orange for depression # the red color #9e471b for distinction
)

#%% test
# sc.pl.umap(adata,color='c92_PAC+')


#%% ################################# c02 PAC+ and NPS PAC-
mask = (adata.obs['c02_PAC']==1) & (adata.obs[nps_pac_col]==-1)
adata.obs['PAC_c02+_' + nps + '-'] = 0
adata.obs.loc[mask, 'PAC_c02+_' + nps + '-'] = 1
adata.obs[nps + '_PAC-'] = 0
adata.obs.loc[adata.obs[nps_pac_col]==-1, nps + '_PAC-'] = 1
adata.obs['c02_PAC-'] = 0
adata.obs.loc[adata.obs['c02_PAC']==-1, 'c02_PAC-'] = 1

#%%
cat_col = 'leiden'
cats = adata.obs[cat_col].unique()

from matplotlib_venn import venn2
pvalues = {}
for cat in cats:
    # skip those cats that have none or too few PACs
    thres = 100
    tb = adata.obs[adata.obs[cat_col]==cat].groupby(['c02_PAC+', nps + '_PAC-']).size().unstack()
    if ((1 not in tb.index) or (1 not in tb.columns)) or \
        (tb.loc[1].sum() < thres or tb[1].sum(0) < thres) or \
        (tb.loc[1, 1] < thres):
        continue

    # hypergeometric test
    M = adata.obs[adata.obs[cat_col]==cat].shape[0]
    n = (adata.obs[adata.obs[cat_col]==cat]['c02_PAC']==1).sum()
    N = (adata.obs[adata.obs[cat_col]==cat][nps_pac_col]==-1).sum()
    x = (adata.obs[adata.obs[cat_col]==cat]['PAC_c02+_' + nps + '-']==1).sum()
    rv = scipy.stats.hypergeom(M, n, N)
    pvalues[cat] = 1-rv.cdf(x)

    # just by frac NOTE already did this, bottom line: there are two subclusters 23, 8 identifeid as sigfniciant (proportion > 0.5) in addition to hypergeometric resutls. Both are in Oligo_OPALIN  & Oligo_RBFOX1. And they 
    # pvalues[cat] = tb.loc[1,1] / min(tb.loc[1].sum(), tb[1].sum(0))

print(pd.Series(pvalues).sort_values().values)
# remove insignificant cats (p>0.05)
pvalues = {k:v for k,v in pvalues.items() if v < 0.05}

pvs = list(pvalues.values())
pvs_ordered = np.array(pvs)[np.argsort(pvs)] + 1e-100
cats_ordered = np.array(list(pvalues.keys()))[np.argsort(pvs)]
custom_order = [cat for cat in cats_ordered]
sns.barplot(x = - np.log10(pvs_ordered), 
            y = cats_ordered, 
            # figure=plt.figure(figsize=(5, 14)),
            order=custom_order,
            color='purple'); 
plt.xlabel('-log10(p-value)')
plt.show()

#%% ########################### look at specific subcluster
sig_celltype = []
for cat in cats_ordered:
    adata.obs[adata.obs[cat_col]==cat].groupby(['c02_PAC+', nps + '_PAC-']).size().unstack()
    sr = adata.obs.groupby(['subtype', 'leiden']).size().unstack()[cat].sort_values(ascending=False)
    sig_celltype.append(sr.index[0])
print(sig_celltype) # results show Oligo_OPALIN, EN_L2_3_IT_PDGFD are interesting
#%% venn diagram TODO needs changing
cat = '18'
try:
    ad_pac_pos = adata.obs.loc[adata.obs[cat_col] == cat, 'c02_PAC'].value_counts()[1]
except KeyError:
    ad_pac_pos = 0
try:
    nps_pac_neg = adata.obs.loc[adata.obs[cat_col] == cat, nps_pac_col].value_counts()[-1]
except KeyError:
    nps_pac_neg = 0
try:
    ad_nps_pac = adata.obs.loc[adata.obs[cat_col] == cat, 'PAC_c02+_' + nps + '-'].value_counts()[1]
except KeyError:
    ad_nps_pac = 0
venn2(subsets=(ad_pac_pos - ad_nps_pac, nps_pac_neg - ad_nps_pac, ad_nps_pac),
        set_colors=('red', 'orange'))
plt.title(cat)
plt.legend(('AD PAC+', 'NPS PAC-'))
plt.show()

#%% ############################### umap
#%% [markdown]
## Visualization of cell types and PAC scores
#%% [markdown]
### subclass
#%%
psychad_palette = pd.read_csv('/home/che82/athan/PASCode/code/github_repo/figures/PsychAD_color_palette_230921.csv', index_col=0)

class_palette = psychad_palette[psychad_palette['name'].isin(adata.obs['class']) & (psychad_palette.index == 'class')]
class_palette = class_palette.sort_values(by='name')
class_palette = dict(zip(
    class_palette['name'].values,
    class_palette['color_hex'].values))
subclass_palette = psychad_palette[psychad_palette['name'].isin(adata.obs['subclass']) & (psychad_palette.index == 'subclass')]
subclass_palette = subclass_palette.sort_values(by='name')
subclass_palette = dict(zip(
    subclass_palette['name'].values, 
    subclass_palette['color_hex'].values))

#%%
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='X_umap',
    class_col='subtype', 
    class_palette=None,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_subclass_umap.tiff'
)
plot_legend(
    legend_handles=handles,
    legend_labels= labels, #[l.split(':')[0] for l in labels],
    legend_ncol=3,
    save_path='./c02_' + nps + '_ovlp_subclass_legend.pdf'
)

handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='X_umap',
    class_col='subtype', 
    class_palette=None,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_subtype_umap.tiff'
)
plot_legend(
    legend_handles=handles,
    legend_labels= labels, #[l.split(':')[0] for l in labels],
    legend_ncol=3,
    save_path='./c02_' + nps + '_ovlp_subtype_legend.pdf'
)

handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='X_umap',
    class_col='leiden', 
    class_palette=None,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_leiden_umap.tiff'
)

#%%
adata.obs['leiden_sig_cluster_c02+_' + nps + '-'] = 'insignificant'
for sig_cluster in list(pvalues.keys()):
    adata.obs.loc[adata.obs['leiden']==str(sig_cluster), 'leiden_sig_cluster_c02+_' + nps + '-'] = str(sig_cluster)
adata.obs['leiden_sig_cluster_c02+_' + nps + '-'] = adata.obs['leiden_sig_cluster_c02+_' + nps + '-'].astype('category')

print(list(pvalues.keys()))
# sig_leiden_palette = {
#     'insignificant': 'grey',
#     '24': 'purple',
#     '22': '#0000FF',
#     '16': 'green',
#     '8': '#FFFF00',
#     '48': '#FF0000',
#     '45': '#42eff5'
# }

#%%
handles, labels = handles, labels = plot_umap(
    adata,
    umap_key='X_umap',
    class_col='leiden_sig_cluster_c02+_' + nps + '-', 
    class_palette=None,
    text_on_plot=False,
    save_path='./c02_' + nps + '_ovlp_sig_leiden_c02+_' + nps + '-_umap.tiff'
)

plot_legend(
    legend_handles=handles,
    legend_labels= [l.split(':')[1] for l in labels],
    legend_ncol=1,
    save_path='./c02_' + nps + '_ovlp_sig_leiden_c02+_' + nps + '-_legend.pdf'
)

#%% [markdown] 
## PAC  umap
plot_pac_umap(
    adata, 
    umap_key='X_umap', 
    pac_col='c02_PAC_score',
    save_path='./c02_' + nps + '_ovlp_c02_PAC_umap.tiff'
)

plot_pac_umap(
    adata, 
    umap_key='X_umap', 
    pac_col=nps + '_PAC_score',
    save_path='./c02_' + nps + '_ovlp_' + nps + '_PAC_umap.tiff'
)


adata.write_h5ad(DATA_PATH + 'c02_' + nps + '_overlap_only_obs_obsm.h5ad')
adata.obs.to_csv(DATA_PATH + 'c02_' + nps + '_overlap.csv')


############for screen blood inflammation markers
#origninal code to generate figures in the paper: 0928_ch_inflam_bloodmarker.py
#here below is just a copy for the code

####################
#%%
%reload_ext autoreload
%autoreload 2
import os

os.chdir('/home/che82/athan/PASCode/code/github_repo/')

from PASCode import *
import seaborn as sns
set_seed(0)

###gene analysis##############
from bioinfokit import analys, visuz
from matplotlib import pyplot as plt
import numpy as np
#%% read in data
cinfo = pd.read_csv("/home/che82/data/psychAD/metadata_latest_0828.csv", index_col=0)
adata0 = sc.read_h5ad('/media/che82/hechenfon/c92x-all_protein_coding.h5ad')


#%% DEG analysis within contrast PAC 1 vs PAC -1 for subclass!!!!!!!!
#PART 1
#sctps = ['Astro','Oligo','OPC','IN_ADARB2','IN_SST','Micro']
sctps = adata0.obs['subclass'].unique()#[1:3]

sgss = ('SAA1','TNF','IL1B','IL6','IL10','IL12A','IL18','IFNG')#,'NPTX2')
tsdfs = []
for sctp in sctps:
    print('=================='+sctp)
    #sctp = 'Micro' #selected cell type
    # this step update the base of log transformation to 2 for accurate calculation of fold change. 
    adata = adata0[adata0.obs['subclass']==sctp].copy()  #NOTE - this is for astrocytes only
    #adata = adata0[(adata0.obs['subtype']=='Astro') & (adata0.obs['c125_68v68_mask']==True)]  #NOTE - this is for astrocytes only
    adata.raw = adata
    sc.pp.log1p(adata)
    adata.X = adata.raw.X
    adata.uns['log1p']['base'] = 2
    #sc.pp.filter_genes(adata, min_cells=100)

    #assign PACs based on GAT score
    adata.obs['pac_pred'] = assign_pac(adata.obs['c92_model_pred_score'], mode='cutoff', cutoff=0.5)
    adata.obs['pac_pred_cat'] = ["pac_"+str(adata.obs.iloc[i]['pac_pred']) for i in range(adata.shape[0])]    #use pac_pred

    adata.obs.loc[(adata.obs['c92x']=='Control') & (adata.obs['pac_pred_cat']=='pac_1.0'),'pac_pred_cat'] = 'nan'
    adata.obs.loc[(adata.obs['c92x']=='Depression_Mood') & (adata.obs['pac_pred_cat']=='pac_-1.0'),'pac_pred_cat'] = 'nan'

    print(adata.obs.groupby(['pac_pred_cat','c92x']).size().unstack())
    tab = adata.obs.groupby('pac_pred_cat').size()
    if (('pac_1.0' in tab.index.values) & ('pac_-1.0' in tab.index.values)) :
        if ((tab['pac_-1.0']>100) & (tab['pac_-1.0'] > 100)) :
            #rank and test for DEGs
            sc.tl.rank_genes_groups(adata, 'pac_pred_cat',groups=['pac_1.0'], reference = 'pac_-1.0',method='wilcoxon',key_added='pac_level') #1 vs -1
            #sc.tl.rank_genes_groups(adata, 'pac_pred_cat',groups=['pac_1.0'], reference = 'rest',method='wilcoxon',key_added='pac_level')  #1 vs all, used this one
            sc.pl.rank_genes_groups_dotplot(adata,n_genes=50, key='pac_level')
            df_pac_phenotype = sc.get.rank_genes_groups_df(adata, group=['pac_1.0'],key='pac_level')
            df0 = df_pac_phenotype.copy() #copy for analysis
            sdf = df0.loc[df0['names'].isin(sgss),:]
            sdf['logMinusP'] = -np.log10(sdf['pvals_adj'])
            sdf['celltype'] = sctp
            print(sdf)
            #save for plotting
            tsdfs.append(sdf.iloc[np.argsort(sdf['names'])])

result = pd.concat(tsdfs, ignore_index=True)
result.to_csv('/home/che82/project/psychAD/fig4/new..method2-subclass_VS_inflam_markers.csv')

#PART 2###for violinplot/umap etc#######################

#%% DEG analysis within contrast PAC 1 vs PAC -1 for selected subclass!!!!!!!! 
###############DEGs()
sctps = ['Astro']#,'Oligo','OPC','IN_ADARB2','IN_SST','Micro']
#sctps = ['EN_L2_3_IT']
for sctp in sctps:
    print('=================='+sctp)
    #sctp = 'Micro' #selected cell type
    # this step update the base of log transformation to 2 for accurate calculation of fold change. 
    adata = adata0[adata0.obs['subclass']==sctp]  #NOTE - this is for astrocytes only
    #adata = adata0[(adata0.obs['subclass']=='Astro') & (adata0.obs['c125_68v68_mask']==True)]  #NOTE - this is for astrocytes only
    adata.raw = adata
    sc.pp.log1p(adata)
    adata.X = adata.raw.X
    adata.uns['log1p']['base'] = 2
    #sc.pp.filter_genes(adata, min_cells=100)

    #predict PACs based on GAT score
    adata.obs['pac_pred'] = assign_pac(adata.obs['c92_model_pred_score'], mode='cutoff', cutoff=0.5)
    adata.obs['pac_pred_cat'] = ["pac_"+str(adata.obs.iloc[i]['pac_pred']) for i in range(adata.shape[0])]    #use pac_pred
    #adata.obs['pac_pred_cat'] = ["pac_"+str(adata.obs.iloc[i]['combined_pac']) for i in range(adata.shape[0])] #use combined pac

    adata.obs.loc[(adata.obs['c92x']=='Control') & (adata.obs['pac_pred_cat']=='pac_1.0'),'pac_pred_cat'] = 'nan'
    adata.obs.loc[(adata.obs['c92x']=='Depression_Mood') & (adata.obs['pac_pred_cat']=='pac_-1.0'),'pac_pred_cat'] = 'nan'
    print(adata.obs.groupby(['pac_pred_cat','c92x']).size().unstack())

    ###for visualization of gene expression
    adata.obsm['X_umap'] = adata.X[:,0:2].toarray()
    adata.obsm['X_umap'][:,0] = adata.obs['umap0'].values
    adata.obsm['X_umap'][:,1] = adata.obs['umap1'].values
    #[adata.obs['umap0'].values,adata.obs['umap1'].values]
    sadata = adata[adata.obs['pac_pred_cat'].isin(['pac_1.0','pac_-1.0'])]
    sc.pl.umap(sadata,color=['pac_pred_cat','IL6','IL12A','IL18','combined_pac'],size=100,)
    sns.histplot(x=sadata.X[:,np.where(sadata.var['gene_name']=='IL18')[0]].toarray().flatten(),hue=sadata.obs['pac_pred_cat'],bins=100,log_scale=[False,True]);plt.show()
    sadata2 = sadata[sadata.X[:,np.where(sadata.var['gene_name']=='IL18')[0]]>0.1]
    sns.ecdfplot(x=sadata2.X[:,np.where(sadata2.var['gene_name']=='IL18')[0]].toarray().flatten(),hue=sadata2.obs['pac_pred_cat']);plt.show()

    my_pal = {"pac_-1.0": "#9e471b",  "pac_1.0": "#0e38c2"}
    vlnp = sns.violinplot(x=sadata2.obs['pac_pred_cat'],y=sadata2.X[:,np.where(sadata2.var['gene_name']=='IL18')[0]].toarray().flatten(),
                   order=['pac_-1.0','pac_1.0'],palette=my_pal)#;plt.show()
    #vlnp.figure.savefig('/home/che82/project/psychAD/fig4/c92_Astro_IL18_violin.pdf')
    #vlnp.figure.savefig('/home/che82/project/psychAD/fig4/c92_EN_L2_3_IT_IL18_violin.pdf')

    ####################
    #rank and test for DEGs
    sc.tl.rank_genes_groups(adata, 'pac_pred_cat',groups=['pac_1.0'], reference = 'pac_-1.0',method='wilcoxon',key_added='pac_level')
    sc.pl.rank_genes_groups_dotplot(adata,n_genes=50, key='pac_level')
    df_pac_phenotype = sc.get.rank_genes_groups_df(adata, group=['pac_1.0'],key='pac_level')
    df0 = df_pac_phenotype.copy() #copy for analysis

    ###gene analysis##############
    from bioinfokit import analys, visuz
    from matplotlib import pyplot as plt
    sgss = ('SAA1','TNF','IL1B','IL6','IL10','IL12A','IL18','IFNG')
    #df0 = df0.loc[np.abs(df0['logfoldchanges'])<1,:]
    visuz.GeneExpression.volcano(df=df0, lfc='logfoldchanges', pv='pvals_adj',lfc_thr=(0.1, 0.1),pv_thr=(1e-2, 1e-2), geneid='names', genenames=sgss,show=True)
    sdf = df0.loc[df0['names'].isin(sgss),:]
    f,axes = plt.subplots(1,2,figsize=(10,3))
    sdf['logMinusP'] = -np.log10(sdf['pvals_adj'])
    sns.barplot(x='names',y='logfoldchanges',data=sdf.iloc[np.argsort(sdf['names']),:],ax=axes[0]).set(title=sctp)
    sns.barplot(x='names',y='logMinusP',data=sdf.iloc[np.argsort(sdf['names']),:],ax=axes[1])
    plt.show
    print(sdf)









# #%%####################HCF gene analysis#####################
# DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

# cinfo = pd.read_csv("/home/che82/data/psychAD/metadata_latest_0828.csv", index_col=0)
# cdata = sc.read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/c02_' + nps + '_overlap.h5ad')
# adata0 = sc.read_h5ad('/media/che82/hechenfon/' + nps + 'x-all_protein_coding.h5ad')
# c02 = sc.read_h5ad('/media/che82/hechenfon/c02x-all.protein_coding.h5ad')

# #%%#check whether the DEGs are enriched across braaks stages
# dinfo = pd.read_csv('/home/che82/project/psychAD/fig4/DEX_c02_' + nps + '_cluster19_filtered.csv')
# sgs = dinfo['names'].tolist()
# #sgs = ["CALR","HSPA5","MARCKS","DNAJB9","TMBIM6","ERLIN2","RNF139","UBQLN2","FAM8A1","JKAMP","TMEM33","SELENOS","SELENOK","TMX1","PPP1R15B"]

# adata = adata0[(adata0.obs['subtype']=='Astro_WIF1') & (adata0.obs.index.isin(cdata.obs.index))]
# #adata = c02[(c02.obs['subtype']=='Astro_WIF1')]

# geneset_aucell(adata, geneset_name='cluster19_sgs',geneset=sgs,AUC_threshold=0.01)

# #%% #some PAC analysis
# df = cdata.obs.drop_duplicates(subset=['SubID'])
# df['c28x'] = cinfo.loc[df['SubID'], 'c28x'].values
# df['c92x'] = cinfo.loc[df['SubID'], 'c92x'].values

# ######on cluster 19 percentage##########
# adata0.obs['leiden'] = np.nan
# adata0.obs.loc[cdata.obs.index, 'leiden'] = cdata.obs['leiden'].values

# ct1 = adata0.obs.groupby(['SubID','leiden']).size().unstack()
# #ct2 = adata0.obs.groupby(['SubID','subtype']).size().unstack()
# #ct2 = adata0.obs.groupby(['SubID','subclass']).size().unstack()
# ct2 = adata0.obs.groupby(['SubID']).size()

# perc = ct1.loc[:,'19']/ct2#.loc[:,'Astro']
# df['ct1'] = ct1.loc[df['SubID'],'19'].values
# df['ct2'] = ct2.loc[df['SubID']].values
# df['c19_perc'] = perc.loc[df['SubID']].values
# sns.boxplot(x='c92x', y='ct1', data=df)
# sns.swarmplot(x='c92x', y='ct1', data=df, color=".25")

# ######on cluster 19 aucell scores[based on the DEG genes identified in the next section]##########
# adata0.obs['cluster19_sgs_aucell'] = np.nan
# adata0.obs.loc[adata.obs.index, 'cluster19_sgs_aucell'] = adata.obs['cluster19_sgs_aucell'].values

# intinfo = adata0.obs[adata0.obs['leiden']=='19']
# summary = intinfo.groupby('SubID')['cluster19_sgs_aucell'].mean()#.reset_index()

# df['cluster19_sgs_aucell'] = summary.loc[df['SubID']].values
# sns.boxplot(x='c92x', y='cluster19_sgs_aucell', data=df)
# sns.swarmplot(x='c92x', y='cluster19_sgs_aucell', data=df, color=".25")

# #########
# df2 = df[df['ct1']>100]
# df2['cluster19_sgs_aucell'] = summary.loc[df2['SubID']].values
# sns.boxplot(x='c92x', y='cluster19_sgs_aucell', data=df2)
# sns.swarmplot(x='c92x', y='cluster19_sgs_aucell', data=df2, color=".25")
# mannwhitneyu(df2[df2['c92x']=='Depression_Mood']['cluster19_sgs_aucell'],df2[df2['c92x']=='Control']['cluster19_sgs_aucell'])

# #%% #####some test######play to see##
# df = adata.obs.drop_duplicates(subset=['SubID'])
# df['c28x'] = cinfo.loc[df['SubID'], 'c28x'].values
# df['c92x'] = cinfo.loc[df['SubID'], 'c92x'].values

# ct1 = adata.obs.groupby(['SubID','subtype']).size().unstack()
# df['ct1'] = ct1.loc[df['SubID'],'Astro_WIF1'].values

# df2 = df[df['ct1']>300]
# intinfo = adata.obs
# summary = intinfo.groupby('SubID')['cluster19_sgs_aucell'].sum()#.reset_index()

# df2['cluster19_sgs_aucell'] = summary.loc[df2['SubID']].values
# sns.boxplot(x='c02x', y='cluster19_sgs_aucell', data=df2)
# sns.swarmplot(x='c02x', y='cluster19_sgs_aucell', data=df2, color=".25")

# #%% #PAC 1 to control
# adata = adata0[(adata0.obs['subtype']=='Astro_WIF1') & (adata0.obs.index.isin(cdata.obs.index))]  #NOTE - this is for astrocytes only
# adata.obs[['leiden','c02_PAC+','c92_PAC+']] = cdata.obs[['leiden','c02_PAC+','c92_PAC+']].loc[adata.obs.index]
# #adata = adata0[(adata0.obs['subclass']=='Astro') & (adata0.obs['c125_68v68_mask']==True)]  #NOTE - this is for astrocytes only
# adata.raw = adata
# sc.pp.log1p(adata)
# adata.X = adata.raw.X
# adata.uns['log1p']['base'] = 2
# #adata.obs['pac_pred'] = assign_pac(adata.obs['c92_model_pred_score'], mode='cutoff', cutoff=0.5)

# ## adding a new obs variable to compare pac 1 vs control
# #df1 = np.array(sc.get.obs_df(adata, ['pac_pred']))
# #df2 = np.array(sc.get.obs_df(adata,['c92x']))
# #adata.obs['pac_c92x'] = adata.obs['c92x'].astype('str')
# #cells = adata.obs_names[np.intersect1d(np.flatnonzero(df1 == 1), np.flatnonzero(df2 == 'Depression_Mood'))]
# #adata.obs.loc[cells,'pac_c92x'] = 'pac1'

# adata.obs['int_leiden'] = adata.obs['leiden'].astype('str')
# #adata.obs['int_leiden'][(adata.obs['int_leiden'] == '19') & (adata.obs['c02_PAC+']==1) & (adata.obs['c92_PAC+']==1)] = 'cluster19'
# adata.obs['int_leiden'][adata.obs['int_leiden'] == '19'] = 'cluster19'
# adata.obs['int_leiden'][adata.obs['int_leiden'] != 'cluster19'] = 'other'

# print(np.unique(adata.obs['int_leiden'],return_counts=True))
# sc.pp.filter_genes(adata, min_cells=10)

# #print(adata.obs.groupby(['c92x', 'pac_c92x']).size().unstack())
# sc.tl.rank_genes_groups(adata, 'int_leiden',groups=['cluster19'], reference = 'other',method='wilcoxon',key_added='pac_donorPhenotype',use_rep='X')
# sc.pl.rank_genes_groups_dotplot(adata,n_genes=50, key='pac_donorPhenotype')

# df_pac_phenotype = sc.get.rank_genes_groups_df(adata, group=['cluster19'],key='pac_donorPhenotype')
# sc.pl.stacked_violin(adata, df_pac_phenotype[:25]['names'].values, groupby='int_leiden', dendrogram=True)

# #getCellFractions(adata,groupby='pac_c92x', geneIDs=adata.var_names).to_csv('results/DEX_pac_donor_%s_percentage.csv'%file)

# df_pac_phenotype.to_csv('/home/che82/project/psychAD/fig4/DEX_c02_c92_cluster19.csv')

# ###sortingAndFilteringDEGs
# pval_adj_lim = 1e-15; logfoldchanges_lim = 0.8 #NOTE threshold

# df = pd.read_csv('/home/che82/project/psychAD/fig4/DEX_c02_c92_cluster19.csv')
# print('orig df shape - %d'%df.shape[0])
# df = df[df['pvals_adj'] < pval_adj_lim]
# df = df[abs(df['logfoldchanges']) > logfoldchanges_lim]
# print('new df shape - %d'%df.shape[0])
# df = df.sort_values(by = 'logfoldchanges',ascending = False)


# df = df#[df['names'].isin(dfp['featurekey'])]
# print('new df shape - %d'%df.shape[0])
# df = df.sort_values(by = 'logfoldchanges',ascending = False)
# df.to_csv('/home/che82/project/psychAD/fig4/DEX_c02_c92_cluster19_filtered.csv')



#%% ##### HCF
# #%%
# ##############################################################################
# sadata = adata[adata.obs['subclass']=='Astro']
# #sc.pp.scale(sadata)
# #sc.tl.pca(sadata)
# sc.pp.neighbors(sadata, n_neighbors=5,use_rep='X_pca_regressed_harmony')
# sc.tl.umap(sadata)

# sadata.obs['leiden_19'] = sadata.obs['leiden_19'].astype('str')

# sadata.obs['leiden_19_c02_pos'] = 0
# mask = (sadata.obs['leiden_19'] == 'True') & (sadata.obs['c02_PAC+'] == 1)
# sadata.obs['leiden_19_c02_pos'][mask] = 1

# sadata.obs['leiden_19_c92_pos'] = 0
# mask = (sadata.obs['leiden_19'] == 'True') & (sadata.obs['c92_PAC+'] == 1)
# sadata.obs['leiden_19_c92_pos'][mask] = 1

# sadata.obs['leiden_19_c02_c92_common'] = 0
# mask = (sadata.obs['leiden_19'] == 'True') & (sadata.obs['PAC_c02+_c92++'] == 1)
# sadata.obs['leiden_19_c02_c92_common'][mask] = 1

# sc.pl.umap(sadata, color=['leiden_19_c02_pos', 'leiden_19_c92_pos','leiden_19_c02_c92_common','leiden'])

# # %%
# ##############################################################################
# adata.obs['leiden_29'] = adata.obs['leiden']=='29'
# sadata = adata[adata.obs['subclass']=='Oligo']
# sc.pp.scale(sadata)
# sc.tl.pca(sadata)
# sc.pp.neighbors(sadata, n_neighbors=50,use_rep='X_pca')
# #sc.pp.neighbors(sadata, n_neighbors=50,use_rep='X_pca_regressed_harmony')
# sc.tl.umap(sadata)

# sadata.obs['leiden_29'] = sadata.obs['leiden_29'].astype('str')

# sadata.obs['leiden_29_c02_pos'] = 0
# mask = (sadata.obs['leiden_29'] == 'True') & (sadata.obs['c02_PAC+'] == 1)
# sadata.obs['leiden_29_c02_pos'][mask] = 1

# sadata.obs['leiden_29_c92_pos'] = 0
# mask = (sadata.obs['leiden_29'] == 'True') & (sadata.obs['c92_PAC+'] == 1)
# sadata.obs['leiden_29_c92_pos'][mask] = 1

# sadata.obs['leiden_29_c02_c92_common'] = 0
# mask = (sadata.obs['leiden_29'] == 'True') & (sadata.obs['PAC_c02+_c92++'] == 1)
# sadata.obs['leiden_29_c02_c92_common'][mask] = 1

# sc.pl.umap(sadata, color=['leiden_29_c02_pos', 'leiden_29_c92_pos','leiden_29_c02_c92_common','leiden'])

#   # %%

#%% HCF count for Chirag
tdata = sc.read_h5ad('/home/che82/Downloads/c92x_Astro_pacMinus.h5ad')

############################

#%%
# import venn3
from matplotlib_venn import venn3
c90 = suball[~suball['c90x'].isna()]
c91 = suball[~suball['c91x'].isna()]
c92 = suball[~suball['c92x'].isna()]

suball.groupby(['c90x', 'c91x']).size().unstack()
suball.groupby(['c90x', 'c92x']).size().unstack()
suball.groupby(['c91x', 'c92x']).size().unstack()
suball.groupby(['c90x', 'c91x', 'c92x']).size().unstack()

venn3(c90.index, c91.index, c92.index)

pos1 = suball[~suball['c90x'].isna()][suball[~suball['c90x'].isna()]['c90x'] !='Control'].index
pos2 = suball[~suball['c91x'].isna()][suball[~suball['c91x'].isna()]['c91x'] !='Control'].index
pos3 = suball[~suball['c92x'].isna()][suball[~suball['c92x'].isna()]['c92x'] !='Control'].index
neg1 = suball[~suball['c90x'].isna()][suball[~suball['c90x'].isna()]['c90x'] =='Control'].index
neg2 = suball[~suball['c91x'].isna()][suball[~suball['c91x'].isna()]['c91x'] =='Control'].index
neg3 = suball[~suball['c92x'].isna()][suball[~suball['c92x'].isna()]['c92x'] =='Control'].index

len(np.intersect1d(pos1, pos2))
len(np.intersect1d(pos1, pos3))
len(np.intersect1d(pos2, pos3))
len(np.intersect1d(pos1, np.intersect1d(pos2, pos3)))

setdiff1 = np.setdiff1d(np.intersect1d(pos1, pos2), np.intersect1d(pos1, np.intersect1d(pos2, pos3)))
len(setdiff1)
setdiff2 = np.setdiff1d(np.intersect1d(pos2, pos3), np.intersect1d(pos1, np.intersect1d(pos2, pos3)))
len(setdiff2)
setdiff3 = np.setdiff1d(np.intersect1d(pos1, pos3), np.intersect1d(pos1, np.intersect1d(pos2, pos3)))
len(setdiff3)



len(neg3)
len(np.intersect1d(neg1, neg2))
len(np.intersect1d(neg1, neg3))
len(np.intersect1d(neg2, neg3))
len(np.intersect1d(neg1, np.intersect1d(neg2, neg3)))

setdiff1 = np.setdiff1d(np.intersect1d(neg1, neg2), np.intersect1d(neg1, np.intersect1d(neg2, neg3)))
len(setdiff1)
setdiff2 = np.setdiff1d(np.intersect1d(neg2, neg3), np.intersect1d(neg1, np.intersect1d(neg2, neg3)))
len(setdiff2)
setdiff3 = np.setdiff1d(np.intersect1d(neg1, neg3), np.intersect1d(neg1, np.intersect1d(neg2, neg3)))
len(setdiff3)


# pos
v = venn3(
    subsets = (5, 83, 20, 7, 5, 55, 33), 
    set_labels = ('Sleep_WeightGain_Guilt_Suicide', 'WeightLoss_PMA', 'Depression_Mood')
)
# neg
v = venn3(
    subsets = (49, 2, 2, 21, 63, 5, 21), 
    set_labels = ('c90 Control', 'c91 Control', 'c92 Control')
)


#%%
###############################################################################
# cell number overlapping venn diagram for 4F,G
###############################################################################
sig_clusters = ['29', '19', '6', '4', '44']

for cat in sig_clusters:
    ad_pac_pos = adata.obs[adata.obs['leiden'] == cat]['c02_PAC+'].value_counts()[1]
    nps_pac_pos = adata.obs[adata.obs['leiden'] == cat]['c92_PAC+'].value_counts()[1]
    ad_nps_pac_pos = adata.obs[adata.obs['leiden'] == cat]['c02_c92_common_PAC+'].value_counts()[1]

    venn2(subsets=(ad_pac_pos - ad_nps_pac_pos, nps_pac_pos - ad_nps_pac_pos, ad_nps_pac_pos),
            set_colors=('red', 'orange'))
    plt.title(cat)
    plt.legend(('AD PAC+', 'NPS PAC+'), loc='lower center')
    plt.savefig(f'/home/che82/athan/PASCode/code/github_repo/figures/s4/cell_num_overlapping_leiden_venn_cluster_{cat}.pdf', dpi=600)
    plt.show()

# %%


























# %%








