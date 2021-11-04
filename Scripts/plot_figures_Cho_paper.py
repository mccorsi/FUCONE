##
from moabb.datasets import (
    Cho2017,
)
from moabb.paradigms import LeftRightImagery
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)

import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd
import seaborn as sns
import os
import numpy as np

if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
if os.path.basename(os.getcwd()) == "Scripts":
    os.chdir("../Database")
basedir = os.getcwd()

## specific functions
def _plot_results_compute_dataset_statistics(stats, filename):
    P, T = find_significant_differences(stats)

    plt.style.use("dark_background")
    columns = stats["pipe1"].unique()
    rows = stats["pipe2"].unique()
    pval_heatmap = pd.DataFrame(columns=columns, index=rows, data=P)
    tval_heatmap = pd.DataFrame(columns=columns, index=rows, data=T)

    mask = np.invert(np.tril(pval_heatmap < 0.05))
    mask = mask[1:, :-1]
    tval_heatmap_2 = tval_heatmap.iloc[1:, :-1].copy()
    vmin = -max(abs(tval_heatmap_2.min().min()), abs(tval_heatmap_2.max().max()))
    vmax = max(abs(tval_heatmap_2.min().min()), abs(tval_heatmap_2.max().max()))

    sns.set ( font_scale=1.8 )
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(24, 18))
        ax = sns.heatmap(
            tval_heatmap_2,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="vlag",
            linewidths=1,
            vmin=vmin,
            vmax=vmax,
            # vmin=0, vmax=0.05,
            cbar_kws={"label": "signif. t-val (p<0.05)"},
        )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.6, top - 0.6)
        ax.set_facecolor("black")
        # ax.xaxis.label.set_size(15)
        # ax.yaxis.label.set_size(15)
        plt.savefig(filename + ".pdf", dpi=300)

def _plot_rainclouds(
    df_results, list_best, list_least, path_figures_root, filename,palette='colorblind'):
    plt.style.use("dark_background")
    ort = "h"
    pal = palette
    sigma = 0.2
    dx = "pipeline"
    dy = "score"
    dhue = "pipeline"
    f, ax = plt.subplots(figsize=(33, 33))
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        data=df_results,
        palette=pal,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2,
    )
    ax.get_legend().remove()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(fontsize=36)
    plt.yticks (fontsize=36 )
    plt.ylabel("Pipeline", fontsize=39)
    plt.xlabel("Score", fontsize=39)
    plt.savefig(path_figures_root + filename + ".pdf", dpi=300)

    f, ax = plt.subplots(figsize=(33, 33))
    df_results_best = df_results[df_results["subject"].isin(list_best)]
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        data=df_results_best,
        palette=pal,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2,
    )
    ax.get_legend().remove()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(fontsize=36)
    plt.yticks (fontsize=36 )
    plt.ylabel("Pipeline", fontsize=39)
    plt.xlabel("Score", fontsize=39)
    plt.savefig(path_figures_root + filename + "_best.pdf", dpi=300)

    f, ax = plt.subplots(figsize=(33, 33))
    df_results_least = df_results[df_results["subject"].isin(list_least)]
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        data=df_results_least,
        palette=pal,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2,
    )
    ax.get_legend().remove()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(fontsize=36)
    plt.yticks (fontsize=36 )
    plt.ylabel("Pipeline", fontsize=39)
    plt.xlabel("Score", fontsize=39)
    plt.savefig(path_figures_root + filename + "_least.pdf", dpi=300)

## dictionary that contains the colors associated to each case studied in the paper
dict_colors={ "RegCSP+shLDA":"#0072B2", "CSP+optSVM": '#B89ED5', 'FgMDM':'#314a93',
              'cov': '#009E73',
             "ensemble-noDR_2":'#1389E6',
              "ensemble-noDR_best": "#BE4E4E",
            'ensemble-noDR_3':'#F97738',
            'ensemble-noDR_4': '#ffd02d',
             "instantaneous":"#576d64", "imcoh":"#c4d5a8",
             "plv":"#64405a" , "pli":"#9e7089", "wpli2_debiased":"#ad96a9","aec":"#3C648E",
              "delta": "#f16745" ,
              "theta": "#ffc65d" ,
              "alpha": "#7bc8A4" ,
              "beta": "#4cc3d9" ,
              "gamma": "#93648d" ,
              "defaultBand": "#F98790",
              }

###################### PART 1 - Cho2017 dataset results ####################
d = Cho2017()
subj = [14, 43, 50, 35, 3, 29, 7, 17, 40, 38]
# list of the most and the least responsive subjects (performance obtained from the cov+elasticnet approach, considered here as our baseline)
list_best = [14, 43, 50, 35, 3]
list_least = [38, 40, 17, 7, 29]
os.chdir(basedir)

path_csv_root = basedir + "/1_Dataset-csv/" + d.code + "/"
path_data_root = basedir + "/2_Dataset-npz/" + d.code + "/"
path_figures_root = basedir + "/0_Figures/" + d.code + "/"

## FC metrics comparison
freqbands = {
    "defaultBand": [8, 35],
}
paradigm = LeftRightImagery(fmin=8, fmax=35)
ch_labels = dict()  # dict that contains all the channel labels
all_res_temp = pd.DataFrame()
plt.close("all")
# load results
res = pd.read_csv(
    path_csv_root
    + "res_np_single_pipelines_Cho2017_preSelectedSubj_OptFCMetrics_defaultBand.csv"
)
all_res_temp = pd.concat([all_res_temp, res])
results_pyr = pd.read_csv(
    path_csv_root
    + "res_np_single_pipelines_Cho2017_preSelectedSubj_OptFCMetrics-pyriemann_0,75.csv"
)
all_res_temp = pd.concat((all_res_temp, results_pyr))

list_FC_paper = [
    "pycohinst-EN",
    "imcoh+elasticnet",
    "plv+elasticnet",
    "pli+elasticnet",
    "wpli2_debiased+elasticnet",
    "aec+elasticnet",
]


all_res_paper = all_res_temp[all_res_temp["pipeline"].isin(list_FC_paper)]
all_res_best_paper = all_res_paper[all_res_paper["subject"].isin(list_best)]
all_res_least_paper = all_res_paper[all_res_paper["subject"].isin(list_least)]

# Create an array with the colors associated to the different types of FC metrics
colors1 = [ dict_colors["instantaneous"] ,
            dict_colors["imcoh"],
        dict_colors["plv"],
        dict_colors["pli"],
        dict_colors["wpli2_debiased"],
        dict_colors["aec"]
           ]
palette1= sns.color_palette(colors1)
stats = compute_dataset_statistics(all_res_paper)
filename = (
    path_figures_root + "Cho2017_Opt_FC-metrics_statcomp_defaultBand"
)
_plot_results_compute_dataset_statistics(stats, filename)

stats_best = compute_dataset_statistics(all_res_best_paper)
filename = (
    path_figures_root
    + "Cho2017_Opt_FC-metrics_statcomp_defaultBand_best"
)
_plot_results_compute_dataset_statistics(stats_best, filename)

stats_least = compute_dataset_statistics(all_res_least_paper)
filename = (
    path_figures_root
    + "Cho2017_Opt_FC-metrics_statcomp_defaultBand_least"
)
_plot_results_compute_dataset_statistics(stats_least, filename)


# plot accuracies
fc_tickslabels=['Instantaneous+EN',"ImCoh+EN","PLV+EN","PLI+EN","wPLI2-d+EN","AEC+EN"]

# group level
plt.style.use("dark_background")#("classic")
f, ax = plt.subplots(figsize=(24, 24))
ax = sns.catplot(
    data=all_res_paper,
    x="pipeline",
    y="score",
    kind="bar",
    palette=palette1,
    order=list_FC_paper,
    height=6, aspect=3
)
plt.ylim((0.4, 1))
plt.xticks(range(len(all_res_paper["pipeline"].unique())),
           fc_tickslabels)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("Pipeline", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.savefig(
    path_figures_root
    + "/all_res_single_pipelines_Cho2017_Pipeline_OptFCMetrics_defaultBand_group.pdf",
    dpi=300,
)

# most responsive subjects
f, ax = plt.subplots(figsize=(24, 24))
ax = sns.catplot(
    data=all_res_best_paper,
    x="pipeline",
    y="score",
    kind="bar",
    palette=palette1,
    order=list_FC_paper,
    height=6, aspect=3
)
plt.ylim((0.4, 1))
plt.xticks(range(len(all_res_paper["pipeline"].unique())),
           fc_tickslabels)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("Pipeline", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.savefig(
    path_figures_root
    + "/all_res_single_pipelines_Cho2017_Pipeline_OptFCMetrics_defaultBand_best.pdf",
    dpi=300,
)

f, ax = plt.subplots(figsize=(24, 24))
ax = sns.catplot(
    data=all_res_least_paper,
    x="pipeline",
    y="score",
    kind="bar",
    palette=palette1,
    order=list_FC_paper,
    height=6, aspect=3
)
plt.ylim((0.4, 1))
plt.xticks(range(len(all_res_paper["pipeline"].unique())),
           fc_tickslabels)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("Pipeline", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.savefig(
    path_figures_root
    + "/all_res_single_pipelines_Cho2017_Pipeline_OptFCMetrics_defaultBand_least.pdf",
    dpi=300,
)

## Frequency bands
spectral_met = [
    "instantaneous+elasticnet",
    "imcoh+elasticnet",
    "plv+elasticnet",
    "wpli2_debiased+elasticnet",
]
fc_tickslabels=['Instantaneous+EN',"ImCoh+EN","PLV+EN","wPLI2-d+EN"]
colors2 = [ dict_colors["delta"] ,
            dict_colors["theta"],
        dict_colors["alpha"],
        dict_colors["beta"],
            dict_colors["gamma"],
            dict_colors["defaultBand"],
           ]
palette2= sns.color_palette(colors2)

results_freqBand_temp = pd.read_csv(
    path_csv_root
    + "res_np_single_pipelines_"
    + d.code
    + "_preSelectedSubj_AllFreq_OptFreqBands_defaultBand.csv"
)
results_freqBand = results_freqBand_temp[
    results_freqBand_temp["pipeline"].isin(spectral_met)
]


plt.style.use("dark_background")#("classic")
# Initialize the figure
f, ax = plt.subplots(figsize=(24, 24))
sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
sns.stripplot(
    x="score",
    y="pipeline",
    hue="FreqBand",
    data=results_freqBand,
    palette=palette2,
    dodge=True,
    alpha=0.30,
    zorder=1,
    size=9,
)
# Show the conditional means, aligning each pointplot in the center of the strips by adjusting the width allotted to each category (.8 by default) by the number of hue levels
sns.pointplot(
    x="score",
    y="pipeline",
    hue="FreqBand",
    data=results_freqBand,
    dodge=0.8 - 0.8 / 6,
    join=False,
    palette=palette2,
    markers="d",
    scale=2.7,
    ci=None,
)
# Improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[6:],
    [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', 'low '+r'$\gamma$', 'Default band'],# labels[6:],  # title="Frequency bands",
    handletextpad=0.5,
    columnspacing=2,
    loc="upper right",
    ncol=1,
    frameon=False,
    bbox_to_anchor=(1.02, 0.5),
    fontsize=24,

)

plt.xlim((0.3,1))
plt.yticks(fontsize=18)
plt.xticks(fontsize=30)
plt.ylabel("Pipeline", fontsize=30)
plt.xlabel("Score", fontsize=30)
plt.savefig(
    path_figures_root + "/Cho2017_Opt_FreqBand.pdf", dpi=300
)

## Pipeline optimization and ensemble
results_ensemble = pd.read_csv(
    path_csv_root + "OptEnsemble-DRnoDR-allsubject-memory_exhaustive_paper.csv"
)
list_fc_ens=['ensemble-noDR_best','ensemble-noDR_2', 'ensemble-noDR_3',
       'ensemble-noDR_4', 'cov+elasticnet', 'instantaneous+elasticnet',
             'imcoh+elasticnet',
       'plv+elasticnet', 'wpli2_debiased+elasticnet',
       ]
colors_fc_ens=[dict_colors['ensemble-noDR_best'],
               dict_colors['ensemble-noDR_2'],
               dict_colors['ensemble-noDR_3'],
               dict_colors['ensemble-noDR_4'],
               dict_colors['cov'], dict_colors['instantaneous'],
               dict_colors['imcoh'],
               dict_colors['plv'], dict_colors['wpli2_debiased'],
               ]
fc_ens_tickslabels=['{Cov,Instant.,ImCoh}',
                    "{Instant.,ImCoh}",
                    "{Instant.,ImCoh,PLV}",
                    "{Instant.,ImCoh,PLV,wPLI2-d}",
                    "Cov","Instantaneous", "ImCoh", "PLV", "wPLI2-d" ]
palette3= sns.color_palette(colors_fc_ens)

list_base_ens=['RegCSP+shLDA', 'CSP+optSVM', 'FgMDM', 'ensemble-noDR_best']
colors_base_ens=[dict_colors['RegCSP+shLDA'],
               dict_colors['CSP+optSVM'],
               dict_colors['FgMDM'],
               dict_colors['ensemble-noDR_best']
               ]
base_ens_tickslabels=['RegCSP+shLDA',
                    'CSP+optSVM',
                    'FgMDM',
                    'FUCONE'
                    ]
palette4= sns.color_palette(colors_base_ens)

# plots FC vs ens
plt.close("all")
plt.style.use("dark_background")
results_ensemble_fc_ens = results_ensemble[
    results_ensemble["pipeline"].isin(list_fc_ens)
]

g = sns.catplot(
    data=results_ensemble_fc_ens,
    x="pipeline",
    y="score",
    kind="bar",
    palette=palette3,
    order=list_fc_ens,
    height=7,
    aspect=4,
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylim((0.4, 1))
plt.xticks(range(len(results_ensemble_fc_ens["pipeline"].unique())),
           fc_ens_tickslabels, fontsize=15)
plt.yticks(fontsize=18)
plt.xlabel("Pipeline", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.savefig(
    path_figures_root + "/Opt_Ensemble_paper_Cho2017_bar_Group_fc_ens.pdf",
    dpi=300,
)

results_ensemble_base_ens = results_ensemble[
    results_ensemble["pipeline"].isin(list_base_ens)
]
results_ensemble_base_ens=results_ensemble_base_ens.replace('ensemble-noDR_best', 'FUCONE')

# plots FUCONE vs state-of-the-art
g = sns.catplot(
    data=results_ensemble_base_ens,
    x="pipeline",
    y="score",
    kind="bar",
    palette=palette4,
    height=4.3,
    aspect=2,
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylim((0.4, 1))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Pipeline", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.savefig(
    path_figures_root + "/Opt_Ensemble_paper_Cho2017_bar_Group_base_ens.pdf",
    dpi=300,
)

filename = "/Opt_Ensemble_paper_Cho2017_raincloud_Group_base_ens"
_plot_rainclouds(
    results_ensemble_base_ens,
    list_best,
    list_least,
    path_figures_root,
    filename,
    palette4
)
