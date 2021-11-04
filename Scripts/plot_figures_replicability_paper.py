## Script to plot results associated with our meta-analysis to assess the replicability of the FUCONE approach

from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)
import moabb.analysis.plotting as moabb_plt

import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd
import numpy as np
import seaborn as sns
import os

if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
if os.path.basename(os.getcwd()) == "Scripts":
    os.chdir("../Database")
basedir = os.getcwd()

## specific functions
def _plot_results_compute_dataset_statistics(stats, filename):
    P, T = find_significant_differences(stats)

    plt.style.use("classic")
    columns = stats["pipe1"].unique()
    rows = stats["pipe2"].unique()
    data = np.array(stats["p"]).reshape((len(rows), len(rows)))
    pval_heatmap = pd.DataFrame(columns=columns, index=rows, data=P)
    tval_heatmap = pd.DataFrame(columns=columns, index=rows, data=T)

    mask = np.invert(np.tril(pval_heatmap < 0.05))
    mask = mask[1:, :-1]
    tval_heatmap_2 = tval_heatmap.iloc[1:, :-1].copy()
    vmin = -max(abs(tval_heatmap_2.min().min()), abs(tval_heatmap_2.max().max()))
    vmax = max(abs(tval_heatmap_2.min().min()), abs(tval_heatmap_2.max().max()))
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
            cbar_kws={"label": "signif. t-val (p<0.05)"},
        )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.6, top - 0.6)
        ax.set_facecolor("white")
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        plt.savefig(filename + "_WithinSession.pdf", dpi=300)

def _plot_rainclouds(df_results, hue_order, path_figures_root, title, filename):
    plt.style.use("dark_background")
    ort = "h"
    pal = "Set2"
    sigma = 0.2
    dx = "pipeline"
    dy = "score"
    dhue = "pipeline"
    f, ax = plt.subplots(figsize=(24, 18))
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        hue_order=hue_order,
        order=hue_order,
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
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(title)
    plt.xlim((0, 1))  # to have always the same scale
    plt.savefig(path_figures_root + filename + "_WithinSession.pdf", dpi=300)

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

list_ppl = [
    "RegCSP+shLDA",
    "CSP+optSVM",
    "FgMDM",
    "cov+elasticnet",
    "ensemble-noDR",
]
list_ppl_ticks=["RegCSP+shLDA",
    "CSP+optSVM",
    "FgMDM",
    "Cov+EN",
    "FUCONE",]
##
os.chdir(basedir)
path_csv_root = basedir + "/1_Dataset-csv/"
path_data_root = basedir + "/2_Dataset-npz/"
path_figures_root = basedir + "/0_Figures/"

plt.style.use("dark_background")

## Replicability - Within-session evaluation
# 001-2014
d = "001-2014"
plt.style.use("dark_background")
path_csv_root_pl = path_csv_root + d + "/"
results_3class = pd.read_csv(
    path_csv_root_pl + "OptEnsemble-coreFC-allsubject-memory-3class.csv"
)
results_3class["dataset"] = results_3class["dataset"].replace(
    "001-2014", "001-2014-3classes"
)
results_2class_rf = pd.read_csv(
    path_csv_root_pl + "OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "001-2014", "001-2014-2classes_rf"
)
results_2class_lhrh = pd.read_csv(
    path_csv_root_pl + "OptEnsemble-coreFC-allsubject-memory-2class-left_hand-right_hand.csv"
)
results_2class_lhrh["dataset"] = results_2class_lhrh["dataset"].replace(
    "001-2014", "001-2014-2classes_lhrh"
)
results_001_2014 = pd.concat((results_3class, results_2class_lhrh, results_2class_rf))


# Schirrmeister2017, 4 classes, left hand, the right hand, both feet, and rest
d = "Schirrmeister2017"
plt.style.use("dark_background")
results_2class_lhrh = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-left_hand-right_hand.csv"
)
results_2class_lhrh["dataset"] = results_2class_lhrh["dataset"].replace(
    "Schirrmeister2017", "Schirrmeister2017-2class_lhrh"
)
results_multiclass = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-4class.csv"
)
results_multiclass["dataset"] = results_multiclass["dataset"].replace(
    "Schirrmeister2017", "Schirrmeister2017-multclasses"
)
results_Schirrmeister = pd.concat((results_2class_lhrh,results_multiclass))


# case Weibo, multi vs 3 classes
d = "Weibo-2014"
plt.style.use("dark_background")
results_2class_rf = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "Weibo-2014", "Weibo-2014-2classes_rf"
)
results_3class = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-DRnoDR-allsubject-memory-3class.csv"
)
results_3class["dataset"] = results_3class["dataset"].replace(
    "Weibo-2014", "Weibo-2014-3classes"
)
results_multiclass = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-DRnoDR-allsubject-memory-multiclass.csv"
)
results_multiclass["dataset"] = results_multiclass["dataset"].replace(
    "Weibo-2014", "Weibo-2014-multclasses"
)
results_Weibo = pd.concat((results_2class_rf, results_3class, results_multiclass))

# Cho2017
d = "Cho2017"
plt.style.use("dark_background")
results_2class = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-DRnoDR-allsubject-memory-2class-left_hand-right_hand.csv"
)
results_2class["dataset"] = results_2class["dataset"].replace(
    "Cho2017", "Cho2017-2classes"
)
results_Cho = results_2class

# Zhou-2016
d = "Zhou-2016"
plt.style.use("dark_background")
results_2class_rf = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "Zhou-2016", "Zhou-2016-2classes_rf"
)
results_2class_lhrh = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-left_hand-right_hand.csv"
)
results_2class_lhrh["dataset"] = results_2class_lhrh["dataset"].replace(
    "Zhou-2016", "Zhou-2016-2classes_lhrh"
)
results_3class = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-3class.csv"
)
results_3class["dataset"] = results_3class["dataset"].replace(
    "Zhou-2016", "Zhou-2016-3classes"
)
results_Zhou = pd.concat((results_2class_lhrh, results_2class_rf, results_3class))

# 001-2015
d = "001-2015"
plt.style.use("dark_background")
results_2class_rf = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "001-2015", "001-2015-2classes_rf"
)
results_001_2015 = results_2class_rf

# 002-2014
d = "002-2014"
plt.style.use("dark_background")
results_2class_rf = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "002-2014", "002-2014-2classes_rf"
)
results_002_2014 = results_2class_rf

# 004-2014
d = "004-2014"
plt.style.use("dark_background")
results_2class_lhrh = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-left_hand-right_hand.csv"
)
results_2class_lhrh["dataset"] = results_2class_lhrh["dataset"].replace(
    "004-2014", "004-2014-2classes_lhrh"
)
results_004_2014 = results_2class_lhrh

# concatenate results from the different datasets, focus on the most important pipelines
results_meta = pd.concat(
    (results_001_2014[results_001_2014["pipeline"].isin(list_ppl)],
        results_001_2015[results_001_2015["pipeline"].isin(list_ppl)],
        results_002_2014[results_002_2014["pipeline"].isin(list_ppl)],
        results_004_2014[results_004_2014["pipeline"].isin(list_ppl)],
        results_Weibo[results_Weibo["pipeline"].isin(list_ppl)],
        results_Schirrmeister[results_Schirrmeister["pipeline"].isin(list_ppl)],
        results_Cho[results_Cho["pipeline"].isin(list_ppl)],
        results_Zhou[results_Zhou["pipeline"].isin(list_ppl)],
    )
)
# to put in evidence our approach
results_meta["pipeline"]=results_meta["pipeline"].replace("ensemble-noDR", "FUCONE")
results_meta["pipeline"]=results_meta["pipeline"].replace("cov+elasticnet", "Cov+EN")

list_2cl = [
    "001-2014-2classes_lhrh",
    "001-2014-2classes_rf",
    "001-2015-2classes_rf",
    "004-2014-2classes_lhrh",
    "002-2014-2classes_rf",
    "Zhou-2016-2classes_lhrh",
    "Zhou-2016-2classes_rf",
    "Schirrmeister2017-2class_lhrh",
    "Weibo-2014-2classes_rf",
]
list_2cl_rf = [
    "001-2014-2classes_rf",
    "001-2015-2classes_rf",
    "002-2014-2classes_rf",
    "Zhou-2016-2classes_rf",
    "Weibo-2014-2classes_rf",
]
list_2cl_lhrh = [
    "001-2014-2classes_lhrh",
    "004-2014-2classes_lhrh",
    "Zhou-2016-2classes_lhrh",
    "Schirrmeister2017-2class_lhrh",
]

results_meta_2class = results_meta[results_meta["dataset"].isin(list_2cl)]
stats = compute_dataset_statistics(results_meta_2class)
for i, pip1 in enumerate(results_meta_2class["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_2class["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_WithinSession/"
                + "Meta_StatComp_AllDatasets_2class_"
                + pip1
                + "_"
                + pip2
                + "_WithinSession.pdf",
                dpi=300,
            )
# stats, 2 classes, rf
results_meta_2class_rf = results_meta[results_meta["dataset"].isin(list_2cl_rf)]
stats = compute_dataset_statistics(results_meta_2class_rf)
for i, pip1 in enumerate(results_meta_2class_rf["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_2class_rf["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_WithinSession/"
                + "Meta_StatComp_AllDatasets_2class_rf_"
                + pip1
                + "_"
                + pip2
                + "_WithinSession.pdf",
                dpi=300,
            )
# stats, 2 classes, lhrh
results_meta_2class_lhrh = results_meta[results_meta["dataset"].isin(list_2cl_lhrh)]
stats = compute_dataset_statistics(results_meta_2class_lhrh)
for i, pip1 in enumerate(results_meta_2class_lhrh["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_2class_lhrh["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_WithinSession/"
                + "Meta_StatComp_AllDatasets_2class_lhrh_"
                + pip1
                + "_"
                + pip2
                + "_WithinSession.pdf",
                dpi=300,
            )

list_3cl = [
    "001-2014-3classes",
    "Weibo-2014-3classes",
    "Zhou-2016-3classes",
]
results_meta_3class = results_meta[results_meta["dataset"].isin(list_3cl)]
stats = compute_dataset_statistics(results_meta_3class)
for i, pip1 in enumerate(results_meta_3class["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_3class["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_WithinSession/"
                + "Meta_StatComp_AllDatasets_3class_"
                + pip1
                + "_"
                + pip2
                + "_WithinSession.pdf",
                dpi=300,
            )


list_mult = [
    "Weibo-2014-multclasses",
    "Schirrmeister2017-multclasses"
]
results_meta_mult = results_meta[results_meta["dataset"].isin(list_mult)]
stats = compute_dataset_statistics(results_meta_mult)
for i, pip1 in enumerate(results_meta_mult["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_mult["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_WithinSession/"
                + "Meta_StatComp_AllDatasets_multclass_"
                + pip1
                + "_"
                + pip2
                + "_WithinSession.pdf",
                dpi=300,
            )


# plot performance distribution over the subjects for each dataset and each pipeline
palette_meta=sns.color_palette([dict_colors["RegCSP+shLDA"],
                           dict_colors["CSP+optSVM"],
                           dict_colors["FgMDM"],
                           dict_colors["cov"],
                           dict_colors["ensemble-noDR_best"],])

plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_2class_rf,
    kind="swarm",
    height=4.2,
    aspect=3,
    dodge=True,
    palette=palette_meta,
)
plt.ylim((0, 1))
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_WithinSession/section_Res_2_Meta_Swarmplot_2class_rf_AllDatasets_WithinSession.pdf",
    dpi=300,
)

plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_2class_lhrh,
    kind="swarm",
    height=4.2,
    aspect=3,
    dodge=True,
    palette=palette_meta,
)
plt.ylim((0, 1))
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_WithinSession/section_Res_2_Meta_Swarmplot_2class_lhrh_AllDatasets_WithinSession.pdf",
    dpi=300,
)

plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_3class,
    kind="swarm",
    height=4.2,
    aspect=1.8,
    dodge=True,
    palette=palette_meta,
)
plt.ylim((0, 1))
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_WithinSession/section_Res_2_Meta_Swarmplot_3class_AllDatasets_WithinSession.pdf",
    dpi=300,
)

plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_mult,
    kind="swarm",
    height=4.2,
    aspect=1.8,
    dodge=True,
    palette=palette_meta,
)
plt.ylim((0, 1))
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_WithinSession/section_Res_2_Meta_Swarmplot_multclass_AllDatasets_WithinSession.pdf",
    dpi=300,
)

## Replicability - Cross-session evaluation

# 001-2014
d = "001-2014"
plt.style.use("dark_background")
path_csv_root_pl = path_csv_root + d + "/"
results_2class_lh_rh = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-left_hand-right_hand.csv"
)
results_2class_lh_rh["dataset"] = results_2class_lh_rh["dataset"].replace(
    "001-2014", "001-2014-2classes_lh_rh"
)
results_2class_rh_f = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-right_hand-feet.csv"
)
results_2class_rh_f["dataset"] = results_2class_rh_f["dataset"].replace(
    "001-2014", "001-2014-2classes_rh_f"
)
results_001_2014 = pd.concat((results_2class_lh_rh, results_2class_rh_f))

# Zhou-2016
d = "Zhou-2016"
plt.style.use("dark_background")
path_csv_root_pl = path_csv_root + d + "/"
results_2class_lh_rh = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-left_hand-right_hand.csv"
)
results_2class_lh_rh["dataset"] = results_2class_lh_rh["dataset"].replace(
    "Zhou-2016", "Zhou-2016-2classes_lh_rh"
)
results_2class_rh_f = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-right_hand-feet.csv"
)
results_2class_rh_f["dataset"] = results_2class_rh_f["dataset"].replace(
    "Zhou-2016", "Zhou-2016-2classes_rh_f"
)
results_Zhou = pd.concat((results_2class_lh_rh, results_2class_rh_f))

# 001-2015
d = "001-2015"
plt.style.use("dark_background")
path_csv_root_pl = path_csv_root + d + "/"
results_2class_rh_f = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-right_hand-feet.csv"
)
results_2class_rh_f["dataset"] = results_2class_rh_f["dataset"].replace(
    "001-2015", "001-2015-2classes_rh_f"
)
results_001_2015 = results_2class_rh_f

# 004-2014 (only cross-session)
d = "004-2014"
plt.style.use("dark_background")
path_csv_root_pl = path_csv_root + d + "/"
results_2class_lh_rh = pd.read_csv(
    path_csv_root_pl
    + "OptEnsemble-coreFC-CrossSession-memory-2class-left_hand-right_hand.csv"
)
results_2class_lh_rh["dataset"] = results_2class_lh_rh["dataset"].replace(
    "004-2014", "004-2014-2classes_lh_rh"
)
results_004_2014 = results_2class_lh_rh

# form a single dataframe with all the results
results_meta = pd.concat(
    (
        results_001_2014[results_001_2014["pipeline"].isin(list_ppl)],
        results_001_2015[results_001_2015["pipeline"].isin(list_ppl)],
        results_004_2014[results_004_2014["pipeline"].isin(list_ppl)],
        results_Zhou[results_Zhou["pipeline"].isin(list_ppl)]
    )
)
# to put in evidence our approach
results_meta["pipeline"]=results_meta["pipeline"].replace("ensemble-noDR", "FUCONE")
results_meta["pipeline"]=results_meta["pipeline"].replace("cov+elasticnet", "Cov+EN")

list_2cl_lh_rh=[ "001-2014-2classes_lh_rh",  "004-2014-2classes_lh_rh", "Zhou-2016-2classes_lh_rh"]
results_meta_lh_rh = results_meta[results_meta["dataset"].isin(list_2cl_lh_rh)]

stats = compute_dataset_statistics(results_meta_lh_rh)
for i, pip1 in enumerate(results_meta_lh_rh["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_lh_rh["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_CrossSession/"
                + "Meta_StatComp_AllDatasets_"
                + pip1
                + "_"
                + pip2
                + "_lh_rh_CrossSession.pdf",
                dpi=300,
            )

list_2cl_rh_f=[ "001-2014-2classes_rh_f", "001-2015-2classes_rh_f", "Zhou-2016-2classes_rh_f"]
results_meta_rh_f = results_meta[results_meta["dataset"].isin(list_2cl_rh_f)]

stats = compute_dataset_statistics(results_meta_rh_f)
for i, pip1 in enumerate(results_meta_rh_f["pipeline"].unique()):
    for k, pip2 in enumerate(results_meta_rh_f["pipeline"].unique()):
        if pip1 != pip2 and i > k:
            moabb_plt.meta_analysis_plot(stats, pip1, pip2)
            plt.savefig(
                path_figures_root
                + "Meta_CrossSession/"
                + "Meta_StatComp_AllDatasets_"
                + pip1
                + "_"
                + pip2
                + "_rh_f_CrossSession.pdf",
                dpi=300,
            )

# plot the performance distribution over the subjects & the datasets
plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_rh_f,
    kind="swarm",
    height=4.2,
    aspect=1.8,
    dodge=True,
    palette=palette_meta,
)
plt.ylim((0, 1))
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_CrossSession/section_Res_2_Meta_Swarmplot_2class_rf_AllDatasets_CrossSession.pdf",
    dpi=300,
)

plt.style.use("dark_background")
sns.catplot(
    x="dataset",
    y="score",
    hue="pipeline",
    hue_order=[
        "RegCSP+shLDA",
        "CSP+optSVM",
        "FgMDM",
        "Cov+EN",
        "FUCONE",
    ],
    data=results_meta_lh_rh,
    kind="swarm",
    height=4.2,
    aspect=1.8,
    dodge=True,
    palette=palette_meta
)
plt.ylim((0, 1))
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Score", fontsize=15)
plt.savefig(
    path_figures_root
    + "Meta_CrossSession/section_Res_2_Meta_Swarmplot_2class_lhrh_AllDatasets_CrossSession.pdf",
    dpi=300,
)


