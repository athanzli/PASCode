import pandas as pd
import time
import rpy2

def _divide_scores(scores, score_col_num, op):
    div_scores = None
    for i in range(score_col_num):
        col_name = 's' + str(i + 1)
        if op == 'pos':
          col = pd.DataFrame({col_name: scores[scores.iloc[:, i] > 0].iloc[:, i]})
        if op == 'neg':
          col = pd.DataFrame({col_name: -1 * scores[scores.iloc[:, i] < 0].iloc[:, i]})
        col['index'] = col.index
        if div_scores is None:
            div_scores = col
        else:
            div_scores = pd.merge(div_scores, col, on='index', how='outer')
    div_scores.index = div_scores['index']
    div_scores.drop('index', axis=1, inplace=True)
    return div_scores

def rra(adata, da_methods=['milo', 'meld', 'daseq']):
    r"""
    
    Returns:
        Aggregated cell labels (adata.obs[aggreglabel_col].values).
    """

    if len(da_methods) == 1 or da_methods is None:
        raise ValueError("Please provide at least two DA methods for aggregation.")

    RobustRankAggreg = rpy2.robjects.packages.importr('RobustRankAggreg')
    rpy2.robjects.pandas2ri.activate()

    scores = adata.obs[da_methods].dropna() 

    print("\n\n----------------------------- RobustRankAggregation started ... -----------------------------")
    st = time.time()

    pscores = _divide_scores(scores, score_col_num=len(da_methods), op='pos')
    nscores = _divide_scores(scores, score_col_num=len(da_methods), op='neg')

    print('Aggregating positive score ranks...')
    r_cp = rpy2.robjects.pandas2ri.py2rpy(pscores)
    pranks = RobustRankAggreg.aggregateRanks(rmat=r_cp, method='RRA')
    pranks = rpy2.robjects.pandas2ri.rpy2py_dataframe(pranks)

    print('Aggregating negative score ranks...')
    r_cn = rpy2.robjects.pandas2ri.py2rpy(nscores)
    nranks = RobustRankAggreg.aggregateRanks(rmat=r_cn, method='RRA')
    nranks = rpy2.robjects.pandas2ri.rpy2py_dataframe(nranks)

    print('Combining positive and negative score ranks...')
    nranks['Score'] = - nranks['Score']
    overlaps = pranks.index.intersection(nranks.index)
    pranks.loc[overlaps, 'Score'] = 0
    nranks.loc[overlaps, 'Score'] = 0
    ranks = pd.concat([pranks, nranks[~nranks.index.isin(overlaps)]])
    rra_col = 'RRA_' + '_'.join(da_methods) # RRA score
    adata.obs.loc[ranks.index, rra_col] = ranks['Score'].values
    adata.obs.loc[adata.obs[rra_col].isna().values, rra_col] = 0
    adata.obs[rra_col] = adata.obs[rra_col].astype(float)

    from .da import assign_pac
    aggreglabel_col = 'AggregLabel_' + '_'.join(da_methods)
    adata.obs[aggreglabel_col] = assign_pac(
        scores=adata.obs[rra_col].values, mode='cutoff', cutoff=0.5)
    adata.obs['aggreg_label'] = adata.obs[aggreglabel_col].values
    print(f"----------------------------- RobustRankAggregation Time cost (s): {(time.time() - st):.2f} -----------------------------\n\n")

    return adata.obs['aggreg_label'].values
