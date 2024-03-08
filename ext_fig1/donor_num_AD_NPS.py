#%%
###############################################################################
# setup
###############################################################################
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.append('../..')
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '/home/che82/athan/ProjectPASCode/data/'

#%%
###############################################################################
# load
###############################################################################
dinfo = pd.read_csv("./dinfo_for_ext_fig1.csv", index_col=0)

# using psychad official palette
palette = pd.read_csv("/home/che82/athan/ProjectPASCode/figures/PsychAD_color_palette_230921.csv", index_col=0)

Na_color = '#e6e6e6'
nps_color = '#0e38c2'
nps_ctl_color = '#addeb3'
color_dic = {
    "AD": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
    "c02x": {"AD": "#591496", 'Na': Na_color, "Control": "#1f7a0f"},
    "SleepWeightGainGuiltSuicide": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "c90x": {"Sleep_WeightGain_Guilt_Suicide": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "WeightLossPMA": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "c91x": {"WeightLoss_PMA": nps_color, 'Na': Na_color, "Control": nps_ctl_color},
    "DepressionMood": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},
    "c92x": {'Depression_Mood': nps_color, 'Na': Na_color, 'Control': nps_ctl_color},

    "c28x": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
    "AD_strict_and_Resilient": {'AD_strict': '#591496', 'Na': Na_color, 'Control': '#1f7a0f', 'AD_resilient': '#20b8da'},
    "Braak stage": {'0.0': "#1f7a0f",
                    '1.0': "#389e26",
                    '2.0': "#6dc95d",
                    '3.0': "#7389d1",
                    '4.0': "#4969d1",
                    '5.0': "#9e61d4",
                    '6.0': "#591496",
                    'Na': Na_color},
    "Braak": {'0.0': "#1f7a0f",
              '1.0': "#389e26",
              '2.0': "#6dc95d",
              '3.0': "#7389d1",
              '4.0': "#4969d1",
              '5.0': "#9e61d4",
              '6.0': "#591496",
              'Na': Na_color},
    "Sex": {'Male': '#40E0D0', 'Na': Na_color, 'Female': '#FF6B00'},
    "Ethnicity": {
        'EUR': '#D81B60',
        'AMR': '#1E88E5',
        'AFR': '#57A860',
        
        'EAS': '#004D40',
        'SAS': '#004D40',
        'EAS_SAS': '#004D40',
        'AS': '#004D40',
        
        'Unknown': '#FFC107'
    },
}

#%%
###############################################################################
# Extended Fig. 1b: donor numbers in AD and NPS
###############################################################################
phenotypes = ['AD', 'SleepWeightGainGuiltSuicide', 'WeightLossPMA', 'DepressionMood']
dinfo = dinfo.rename(columns={
    'c02x':'AD',
    'c90x':'SleepWeightGainGuiltSuicide',
    'c91x':'WeightLossPMA',
    'c92x':'DepressionMood',
    'Braak':'Braak stage'
})

all_cat_colors = {}
for category, value_color_pairs in color_dic.items():
    all_cat_colors.update(value_color_pairs)

for i in range(len(phenotypes)):
    df = dinfo[dinfo[phenotypes[i]]!='Na']
    print('Number of donors:', df.shape[0])
    df = df[[phenotypes[i], 'Braak stage', 'Sex', 'Ethnicity']]
    df_new = {col: df[col].value_counts(normalize=True) for col in df.columns} # NOTE choose normalize
    df_new_nonorm = {col: df[col].value_counts(normalize=False) for col in df.columns} # NOTE choose normalize
    df_new = pd.DataFrame(df_new).T.fillna(0)
    df_new_nonorm = pd.DataFrame(df_new_nonorm).T.fillna(0)

    colors = [all_cat_colors.get(col) for col in df_new.columns]
    if phenotypes[i] != 'AD':
        j = list(df_new.columns).index('Control')
        colors[j] = nps_ctl_color

    df_new_nonorm = df_new_nonorm.astype(int)
    ax = df_new_nonorm.plot(
        kind='bar',
        stacked=True,
        figsize=(6,6),
        color=colors,
        legend=False)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if height > 0:  # or some small threshold instead of 0 to avoid clutter
            
            ax.annotate(f'{int(height)}', (p.get_x() + width / 2, p.get_y() + height * 0.5), ha='center')

    plt.savefig(f"donor_num_stacked_barplot_{phenotypes[i]}.pdf", dpi=600)
    plt.show()


# %%
