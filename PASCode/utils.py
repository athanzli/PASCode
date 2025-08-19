import numpy as np
import pandas as pd
import warnings
import scanpy as sc
from typing import List, Optional
import anndata
import os
import numpy as np

def subject_info(
        obs, 
        subid_col, 
        columns
    ):
    r"""Summarize subject-level information from a single-cell observation dataframe.

    Args:
        obs (pandas.DataFrame): indicies are cell barcode IDs, and columns are
            relevant information such as cell type, subject ID, subject sex,
            subject braak stages, condition labels, etc.
        subid_col (str): column name for subject IDs.
        columns: columns of interest to be summarized.
    Returns:
        subinfo (pandas.DataFrame): summarized subject-level information.
    """
    warnings.filterwarnings("ignore")
    obs[subid_col] = obs[subid_col].astype('category').cat.remove_unused_categories()
    subinfo = pd.DataFrame(index=obs[subid_col].unique(), columns=columns)
    for col in columns:
        df = obs[[subid_col, col]].drop_duplicates()
        subinfo.loc[df[subid_col], col] = df[col].values
    cell_num = obs[subid_col].value_counts()
    subinfo.loc[cell_num.index, 'cell_num'] = cell_num.values
    subinfo.sort_index(axis=0, inplace=True)
    subinfo = subinfo.sort_values(by='cell_num')
    subinfo['cell_num'] = subinfo['cell_num'].astype(int)
    return subinfo


def condition_equal_subjects(adata, subid_col, cond_col):
    r"""
    Requiring binary conditions.
    
    """
    dinfo = subject_info(adata.obs, subid_col, columns=[cond_col])
    min_num = dinfo[cond_col].value_counts().min()
    max_num = dinfo[cond_col].value_counts().max()
    if min_num == max_num:
        return True
    return False

def subsample_donors(
    adata,
    subid_col,
    cond_col,
    pos_cond,
    neg_cond,
    sex_col=None,
    subsample_num=None,
    mode='top', # 'random' or 'top'
):
    r"""
    Subsample donors based on the condition
    
    Args:
        adata (AnnData): Annotated data matrix.
        subsample_num (str): Number of donors to subsample, e.g., '10:10'.
        subid_col (str): Column name for donor ID.
        cond_col (str): Column name for condition.
        pos_cond (str): Positive condition name.
        neg_cond (str): Negative condition name.
        sex_col (str, optional): Column name for sex. Default: None.
        mode (str, optional): Subsampling mode. 'random' means randomly select donors, 'top' means select donors with top cell number. Default: 'random'.

    Returns:
        AnnData: subsampled AnnData object.
    """
    print("Before donor subsampling:")
    dinfo = subject_info(
        obs=adata.obs,
        subid_col=subid_col,
        columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
    print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())

    if subsample_num is None:
        print("'subsample_num' not provided. Automatically subsample to the minimum number of subjects in the two conditions.")
        min_num = dinfo[cond_col].value_counts().min()
        subsample_num = str(min_num) + ':' + str(min_num)

    print("Donor subsampling: ", subsample_num)
    pos_donor_num = int(subsample_num.split(':')[0])
    neg_donor_num = int(subsample_num.split(':')[1])
    if sex_col is None:
        if mode == 'random':
            sel_pos = dinfo[dinfo[cond_col]==pos_cond].sample(n=pos_donor_num).index
            sel_neg = dinfo[dinfo[cond_col]==neg_cond].sample(n=neg_donor_num).index
        elif mode == 'top':
            sel_pos = dinfo[dinfo[cond_col]==pos_cond].sort_values(by='cell_num', ascending=False)[:pos_donor_num].index
            sel_neg = dinfo[dinfo[cond_col]==neg_cond].sort_values(by='cell_num', ascending=False)[:neg_donor_num].index
        chosen_donors = np.concatenate((sel_pos, sel_neg))
    else:
        pos_hf_num = pos_donor_num // 2
        neg_hf_num = neg_donor_num // 2
        df = dinfo.groupby([cond_col, sex_col]).size().unstack()
        female = df.columns[0] # we ignored a strict correspondence here
        male = df.columns[1]
        if df.loc[pos_cond].min() >= pos_hf_num:
            posm_num = pos_hf_num
            posfm_num = pos_hf_num
        else:
            posm_num = df.loc[pos_cond, male]
            posfm_num = pos_donor_num - posm_num
        if df.loc[neg_cond].min() >= neg_hf_num:
            negm_num = neg_hf_num
            negfm_num = neg_hf_num
        else:
            negm_num = df.loc[neg_cond, male]
            negfm_num = neg_donor_num - negm_num
        
        if mode == 'random':
            pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sample(n=posm_num).index
            pos_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sample(n=posfm_num).index
            neg_m_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==male)].sample(n=negm_num).index
            neg_fm_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==female)].sample(n=negfm_num).index
        elif mode == 'top':
            pos_m_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==male)].sort_values(by='cell_num', ascending=True)[:posm_num].index
            pos_fm_donors = dinfo[((dinfo[cond_col]==pos_cond)) & (dinfo[sex_col]==female)].sort_values(by='cell_num', ascending=True)[:posfm_num].index
            neg_m_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==male)].sort_values(by='cell_num', ascending=True)[:negm_num].index
            neg_fm_donors = dinfo[((dinfo[cond_col]==neg_cond)) & (dinfo[sex_col]==female)].sort_values(by='cell_num', ascending=True)[:negfm_num].index
        chosen_donors = np.concatenate((pos_m_donors, pos_fm_donors, neg_m_donors, neg_fm_donors))
    adata = adata[adata.obs[subid_col].isin(chosen_donors)]
    print("After donor subsampling:")
    dinfo = subject_info(
        obs=adata.obs,
        subid_col=subid_col,
        columns=[cond_col, sex_col] if sex_col is not None else [cond_col])
    print(dinfo.groupby([cond_col, sex_col]).size().unstack() if sex_col is not None else dinfo[cond_col].value_counts())

    # remove graph connectivity, if it was in original adata. Otherwise, it will cause error
    if 'neighbors' in adata.uns.keys():
        del adata.uns['neighbors']
    if 'connectivities' in adata.obsp.keys():
        del adata.obsp['connectivities']

    return adata

from .graph import build_graph
def align_psychad_gene(adata: anndata.AnnData,):
    r"""
    Take the intersection of adata.var_names and PsychAD genes, and reorder adata.var_names to match the order of PsychAD genes.

    Returns:
        AnnData: AnnData object with genes aligned to PsychAD genes.
    """
    if 'connectivities' not in adata.obsp.keys():
        print("No graph found in adata. Building graph using PCA...")
        build_graph(adata, run_umap=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "PsychAD_hvg_3401.csv")
    psychad_genes = pd.read_csv(file_path, index_col=0)

    adata_ori = adata.copy()
    ovlp_genes = adata.var_names.intersection(psychad_genes.index)
    rest_genes = psychad_genes.index.difference(ovlp_genes)
    rest_genes_var = pd.DataFrame(index=rest_genes, columns=['gene'], data=rest_genes)
    adata = anndata.AnnData(
        X = np.hstack([adata.X, np.zeros((adata.shape[0], rest_genes.shape[0]))]),
        obs = adata.obs,
        var = pd.concat([adata.var, rest_genes_var]),
    )
    adata = adata[:, psychad_genes.index]
    if len(adata_ori.uns) > 0:
        adata.uns = adata_ori.uns
    if len(adata_ori.obsm) > 0:
        adata.obsm = adata_ori.obsm
    if len(adata_ori.obsp) > 0:
        adata.obsp = adata_ori.obsp
    return adata

def scmatrix(obs: pd.DataFrame, 
             subid_col: str, 
             class_col: str, 
             score_col: str,
             column_order: Optional[List] = None) -> pd.DataFrame:
    r"""
    Subject-celltype matrix.
    """
    warnings.filterwarnings("ignore")
    obs.loc[:, subid_col] = obs[subid_col].astype('category').cat.remove_unused_categories()
    count_scmat = obs.groupby([subid_col, class_col]).size().unstack()
    scmat = obs.groupby([subid_col, class_col])[score_col].sum().reset_index()
    scmat = scmat.pivot(index=subid_col, columns=class_col, values=score_col)
    scmat = scmat.div(count_scmat)
    scmat[scmat.isna()] = 0
    scmat.sort_index(axis=0, inplace=True)
    scmat.sort_index(axis=1, inplace=True)
    if column_order is not None:
        scmat = scmat.loc[:, column_order]
    return scmat

###############################################################################
# plot utilities
###############################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, hex2color, to_hex
from matplotlib.cm import ScalarMappable
def plot_umap(
    adata, 
    umap_key, 
    class_col, 
    class_palette, 
    text_on_plot=True, 
    save_path='./',
    s=2,
    use_default_colors=False,
    s_text=40
):
    r"""
    Args: 
        class_palette: dict
    """

    # print(plt.rcParams['font.family'])
    # print(plt.rcParams['font.style'])
    
    adata.obs[class_col] = adata.obs[class_col].astype('category')

    id_and_class_col = 'id_and_' + class_col
    class_id_col = class_col + '_id'
    
    # dic = dict(zip(adata.obs[class_col].unique().sort_values(), 
    #     [str(int(cl_id) + 1) + ': ' + str(cl)
    #     for cl_id, cl in enumerate(adata.obs[class_col].unique().sort_values())]))
    if class_palette is not None:
        dic = dict(zip(
            list(class_palette.keys()),
            [str(int(i)+1) + ': ' + list(class_palette.keys())[i] for i in range(len(class_palette))]
        ))
    else:
        dic = dict(zip(
            adata.obs[class_col].unique().sort_values(),
            [str(int(i)+1) + ': ' + str(adata.obs[class_col].unique().sort_values()[i]) 
                for i in range(len(adata.obs[class_col].unique().sort_values()))]
        ))
    adata.obs[id_and_class_col] = [dic[cl] for cl in adata.obs[class_col]]
    adata.obs[class_id_col] = [int(dic[cl].split(':')[0]) for cl in adata.obs[class_col]]

    if class_palette is not None:
        # class_palette = dict(sorted(class_palette.items(), key=lambda x: x[0]))
        # id_and_class_palette = dict(zip(
        #     [str(int(i)+1) + ': ' + str(adata.obs[class_col].unique().sort_values()[i]) 
        #         for i in range(len(adata.obs[class_col].unique().sort_values()))],
        #     list(class_palette.values()))
        # )
        id_and_class_palette = dict(zip(
            [str(int(i)+1) + ': ' + list(class_palette.keys())[i] 
                for i in range(len(class_palette))],
            list(class_palette.values()))
        )
    else:
        num_colors = adata.obs[class_col].nunique()
        id_and_class_palette = sns.color_palette("tab20", n_colors=num_colors) # NOTE

    _, ax = plt.subplots(figsize=(15, 15))
    sc.pl.embedding(
        adata, 
        basis=umap_key,
        color=id_and_class_col,
        palette=id_and_class_palette if not use_default_colors else None,
        size=s, # NOTE 
        ax=ax, 
        title=None, 
        show=False, 
        frameon=False
    )
    ax.set_title(None)
    legend_handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    # function from https://github.com/scverse/scanpy/issues/1513
    from adjustText import adjust_text
    def gen_mpl_labels(
        adata, groupby, exclude=(), umap_key='X_umap',
        ax=None, adjust_kwargs=None, text_kwargs=None):
        if adjust_kwargs is None:
            adjust_kwargs = {"text_from_points": False}
        if text_kwargs is None:
            text_kwargs = {}

        medians = {}

        for g, g_idx in adata.obs.groupby(groupby).groups.items():
            if g in exclude:
                continue
            medians[g] = np.median(adata[g_idx].obsm[umap_key], axis=0)

        if ax is None:
            texts = [
                plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
            ]
        else:
            texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
        adjust_text(texts, **adjust_kwargs)

    if text_on_plot:
        gen_mpl_labels(
            adata=adata,
            groupby=class_id_col,
            exclude=("None",),  # This was before we had the `nan` behaviour
            umap_key=umap_key,
            ax=ax,
            adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
            text_kwargs=dict(fontsize=s_text),
        )
    fig = ax.get_figure()
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, format="tiff", bbox_inches="tight", dpi=600)
    plt.show()
    return legend_handles, labels

def plot_pac_score(adata, xy_key, pac_col, colors=None, save_path=None, s=2):
    if colors is None:
        # colors = ['#1f7a0f', '#ffffff', '#591496']
        colors = ['#4e72ba', '#ffffff', '#b54a63'] # blue, white, red
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    # norm = plt.Normalize(-1, 1)
    norm = adata.obs[pac_col].min(), adata.obs[pac_col].max()
    plt.figure(figsize=(15, 15))
    sns.scatterplot(
        x=adata.obsm[xy_key][:, 0], 
        y=adata.obsm[xy_key][:, 1], 
        hue=adata.obs[pac_col],
        palette=cmap, 
        s=s,
        hue_norm=norm, 
        legend=False
    )
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if save_path:
        plt.savefig(save_path, format='tiff', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def plot_pac_umap(adata, umap_key, pac_col, colors, save_path, s=2):
    if colors is None:
        colors = ['#1f7a0f', '#ffffff', '#591496']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    norm = plt.Normalize(-1, 1)
    plt.figure(figsize=(15, 15))
    sns.scatterplot(
        x=adata.obsm[umap_key][:, 0], 
        y=adata.obsm[umap_key][:, 1], 
        hue=adata.obs[pac_col],
        palette=cmap, 
        s=s,
        hue_norm=norm, 
        legend=False
    )
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if save_path:
        plt.savefig(save_path, format='tiff', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    # for the color bar
    fig, ax = plt.subplots(figsize=(0.5, 10))
    bounds = np.linspace(-1, 1, 1000)
    ticks = [-1, -0.5, 0, 0.5, 1]
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-1, vmax=1), cmap=cmap),
                    cax=ax, orientation='vertical', boundaries=bounds, ticks=ticks, spacing='proportional')
    cb.ax.tick_params(labelsize=30) # TODO
    cb.set_ticklabels(['-1.0', '-0.5', '0', '0.5', '1.0'])
    if save_path:
        plt.savefig('./PAC_umap_colorbar.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

def plot_legend(legend_handles, legend_labels, legend_ncol, save_path):
    fig = plt.figure(figsize=(3, 1.5))
    ax = fig.add_subplot()
    ax.axis('off')
    legend = ax.legend(legend_handles, legend_labels, loc='center', 
                       fontsize=24, ncol=legend_ncol)
    new_marker_size = 360
    for handle in legend.legendHandles:
        handle._sizes = [new_marker_size]
    for spine in ax.spines.values():
        spine.set_visible(False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()


def distinct_color_scatterplot(x, y, label, s):
    num_colors = len(np.unique(label))
    palette = sns.color_palette("tab20", n_colors=num_colors)
    sns.scatterplot(x=x, y=y, hue=label, s=s, palette=palette, )
    plt.legend(bbox_to_anchor=(0.5, -.2), loc='upper center', ncol=3)
    plt.show()

def plot_cell_scores(x, y, scores, s, colors = ['#73AE65', '#ffffff', '#8C72A6'],
                     save_fig_path=None, save_fig_name=None,
                     set_xlim=None, set_ylim=None, title=''):
    # colors = ["#0000ff", "#ffffff", "#ff0000"] # blue, white, red
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    norm = plt.Normalize(scores.min(), scores.max())
    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(x=x, y=y, hue=scores,
                        palette=cmap, s=s, norm=norm, legend=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if (save_fig_name is None) or (save_fig_path is None):
        plt.show()
    else:    
        plt.savefig(save_fig_path + save_fig_name + '.png', dpi=300)


def plot_color_bar(
    title,
    vmin=0,
    vmax=1,
    st_color='white',
    end_color='purple',
    fontsize=50,
    save_path = None
):

    def interpolate_colors(st_color, end_color, n_steps):
        start_rgb = hex2color(st_color)
        end_rgb = hex2color(end_color)
        
        color_list = [to_hex((start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (n_steps - 1),
                                        start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (n_steps - 1),
                                        start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (n_steps - 1))) 
                    for i in range(n_steps)]
        return color_list

    n_steps = 21
    interpolated_colors = interpolate_colors(st_color, end_color, n_steps)
    color_bar = LinearSegmentedColormap.from_list('MyColorBar', interpolated_colors)

    sm = ScalarMappable(cmap=color_bar, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    fig, ax = plt.subplots(figsize=(3, 8))
    plt.axis('off')
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', aspect=20)
    cbar.set_label(title, labelpad=30, fontsize=fontsize)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.label.set_verticalalignment('center')
    cbar.ax.tick_params(labelsize=fontsize)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()