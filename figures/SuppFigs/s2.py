#%%
import sys
sys.path.append('../..')
import PASCode

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata

PASCode.random_seed.set_seed(0)

DATA_PATH = '/home/che82/athan/PASCode/code/github_repo/data/'

# %%
adata = sc.read_h5ad(DATA_PATH + "/PsychAD/c02_only_obs_obsm.h5ad")
adata_pac = sc.read_h5ad(DATA_PATH + "/PsychAD/c02_100v100_gb_only_obs_obsm.h5ad")

#%%
###############################################################################
# score histogram
###############################################################################
obs = adata_pac.obs
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
methods = ['milo', 'meld', 'cna', 'daseq', 'rra_milo_meld_cna_daseq', 'rra_milo_meld_daseq']
for i in range(2):
    for j in range(3):
        method = methods[i*3+j]
        sns.histplot(obs[method], bins=100, ax=axes[i, j])
        axes[i, j].set_title(method)
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')
plt.tight_layout()
plt.savefig('./c02_100v100_methods_pac_score_hist.pdf', dpi=600)

#%%
obs = adata.obs[adata.obs['c02_100v100_mask']]
palette = {
    1.0: '#591496',
    0.0: 'grey',
    -1.0: '#1f7a0f'
}
sns.histplot(data=obs, x='pac_score', hue='rra_pac', bins=100, palette=palette)
plt.savefig('./c02_100v100_pac_score_rra_score_hist.pdf', dpi=600)
plt.close()


# %%
###############################################################################
# c02 11v11, ROSMAP, SEA-AD UMAP and Latent UMAP
###############################################################################
from PASCode.utils import plot_pac_umap, plot_umap, plot_legend
adata1 = sc.read_h5ad(DATA_PATH + 'PsychAD/c02_only_obs_obsm.h5ad')
# adata2 = sc.read_h5ad(DATA_PATH + 'SEA-AD/seaad.h5ad')
adata2 = sc.read_h5ad(DATA_PATH + 'SEA-AD/seaad_no_gxp.h5ad')
# adata2 = anndata.AnnData(
#     X=None,
#     obs=adata2.obs,
#     var=adata2.var,
#     obsm={'X_umap': adata2.obsm['X_umap']},
#     obsp=adata2.obsp,
# )
# adata2.write_h5ad("./seaad_no_gxp.h5ad")
adata3 = sc.read_h5ad(DATA_PATH + 'ROSMAP/rosmap_ovlpgenes_with_psychAD_contrasts.h5ad')

# adata1.obs['model_layer2_umap0'] = adata1.obsm['model_layer2_umap'][:,0]
# adata1.obs['model_layer2_umap1'] = adata1.obsm['model_layer2_umap'][:,1]
# adata1.obs['umap0'] = adata1.obsm['X_umap'][:,0]
# adata1.obs['umap1'] = adata1.obsm['X_umap'][:,1]

psychad_palette = pd.read_csv('../PsychAD_color_palette_230921.csv', index_col=0)

class_palette = psychad_palette[psychad_palette['name'].isin(adata1.obs['class']) & (psychad_palette.index == 'class')]
class_palette = class_palette.sort_values(by='name')
class_palette = dict(zip(
    class_palette['name'].values,
    class_palette['color_hex'].values))

subclass_palette = psychad_palette[psychad_palette['name'].isin(adata1.obs['subclass']) & (psychad_palette.index == 'subclass')]
subclass_palette = subclass_palette.sort_values(by='name')
subclass_palette = dict(zip(
    subclass_palette['name'].values, 
    subclass_palette['color_hex'].values))

#%% c02 11v11
handles, labels = handles, labels = plot_umap(
    adata1[adata1.obs['c02_11v11_mask']],
    umap_key='X_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./c02_11v11_subclass_umap.tiff',
    s=10
)

handles, labels = handles, labels = plot_umap(
    adata1[adata1.obs['c02_11v11_mask']],
    umap_key='model_layer2_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./c02_11v11_subclass_latent_umap.tiff',
    s=10
)

plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./psychAD_subclass_legend.pdf')

plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='X_umap', 
    pac_col='rra_pac',
    save_path='./c02_11v11_rra_pac_umap.tiff',
    colors=None
)

plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='X_umap', 
    pac_col='pac_score',
    save_path='./c02_11v11_pac_score_umap.tiff',
    colors=None
)

plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='model_layer2_umap', 
    pac_col='rra_pac',
    save_path='./c02_11v11_rra_pac_latent_umap.tiff',
    colors=None
)

plot_pac_umap(
    adata1[adata1.obs['c02_11v11_mask']], 
    umap_key='model_layer2_umap', 
    pac_col='pac_score',
    save_path='./c02_11v11_pac_score_latent_umap.tiff',
    colors=None
)


#%% SEA-AD
handles, labels = handles, labels = plot_umap(
    adata2,
    umap_key='X_umap',
    class_col='Subclass', 
    class_palette=None,
    text_on_plot=True,
    save_path='./seaad_original_subclass_umap.tiff',
    s=10
)

plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./seaad_original_subclass_legend.pdf')

handles, labels = handles, labels = plot_umap(
    adata2,
    umap_key='X_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./seaad_subclass_umap.tiff',
    s=10
)

handles, labels = handles, labels = plot_umap(
    adata2,
    umap_key='model_layer2_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./seaad_subclass_latent_umap.tiff',
    s=10
)

plot_pac_umap(
    adata2, 
    umap_key='X_umap', 
    pac_col='rra_pac',
    save_path='./seaad_rra_pac_umap.tiff',
    colors=None,
    s=10
)

plot_pac_umap(
    adata2, 
    umap_key='X_umap', 
    pac_col='pac_score',
    save_path='./seaad_pac_score_umap.tiff',
    colors=None,
    s=10
)

plot_pac_umap(
    adata2, 
    umap_key='model_layer2_umap', 
    pac_col='rra_pac',
    save_path='./seaad_rra_pac_latent_umap.tiff',
    colors=None,
    s=10
)

plot_pac_umap(
    adata2, 
    umap_key='model_layer2_umap', 
    pac_col='pac_score',
    save_path='./seaad_pac_score_latent_umap.tiff',
    colors=None,
    s=10
)

#%% ROSMAP
adata3.obsm['X_tsne'] = np.hstack([
    adata3.obs['tsne1'].values.reshape(-1, 1),
    adata3.obs['tsne2'].values.reshape(-1, 1)
])
handles, labels = handles, labels = plot_umap(
    adata3,
    umap_key='X_tsne', # NOTE
    class_col='Subcluster', 
    class_palette=None,
    text_on_plot=True,
    save_path='./rosmap_original_subclass_umap.tiff',
    s=20
)
plot_legend(
    legend_handles=handles,
    legend_labels=labels,
    legend_ncol=2,
    save_path='./rosmap_original_subclass_legend.pdf')

handles, labels = handles, labels = plot_umap(
    adata3,
    umap_key='X_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./rosmap_subclass_umap.tiff',
    s=20
)

handles, labels = handles, labels = plot_umap(
    adata3,
    umap_key='model_layer2_umap',
    class_col='subclass', 
    class_palette=subclass_palette, # TODO you need to debug this, if text_on_plot=false the legend is not correct
    text_on_plot=True,
    save_path='./rosmap_subclass_latent_umap.tiff', 
    s=20
)

plot_pac_umap(
    adata3, 
    umap_key='X_umap', 
    pac_col='rra_pac',
    save_path='./rosmap_rra_pac_umap.tiff',
    colors=None,
    s=20
)

plot_pac_umap(
    adata3, 
    umap_key='X_umap', 
    pac_col='pac_score',
    save_path='./rosmap_pac_score_umap.tiff',
    colors=None,
    s=20
)

plot_pac_umap(
    adata3, 
    umap_key='model_layer2_umap', 
    pac_col='rra_pac',
    save_path='./rosmap_rra_pac_latent_umap.tiff',
    colors=None,
    s=20
)

plot_pac_umap(
    adata3, 
    umap_key='model_layer2_umap', 
    pac_col='pac_score',
    save_path='./rosmap_pac_score_latent_umap.tiff',
    colors=None,
    s=20
)



A = np.random.rand(9, 9)

sns.heatmap(A)

