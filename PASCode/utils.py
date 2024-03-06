import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
import scanpy as sc
from typing import List, Optional

def subject_info(
        obs, 
        subid_col, 
        columns
    ):
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
    return subinfo

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

    print("TEMPTMPE!!!!")
    print(plt.rcParams['font.family'])
    print(plt.rcParams['font.style'])
    

    
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


from matplotlib.colors import LinearSegmentedColormap, hex2color, to_hex
from matplotlib.cm import ScalarMappable
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
