## Script to plot results associated with the extension of the FUCONE approach with dimension reduction
from moabb.datasets import (
    Schirrmeister2017,
    BNCI2015004
)
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)

import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from visbrain.objects import TopoObj, ColorbarObj, SceneObj
from collections import Counter

if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
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


def _plot_topo_features_extensive(path_figures_root, dataset, DR_chan_select):
    plt.style.use("dark_background")
    cmap = "magma"
    ch_pos = pd.read_csv(f"{os.getcwd()}/Datasets_montages_{dataset.code}.csv")
    ch_pos.rename(columns={ch_pos.columns[0]: "Label"})
    xy = ch_pos.to_numpy()[:, 1:3]
    xy = xy.astype(np.float)
    ch_names = ch_pos[ch_pos.columns[0]].to_list()
    pipelines = DR_chan_select["pipeline"].unique()
    nd_list=DR_chan_select["n_dr"].unique()
    for n in nd_list:
        for p in pipelines:
            # create a dict that countains how many times a given channel is selected as a feature
            all_res_t = DR_chan_select[
                (DR_chan_select["pipeline"] == p) &
                (DR_chan_select["n_dr"] == n)
            ]
            feat = all_res_t["chan_select"].explode().to_list()
            feat_unique = set(feat)
            feat_count = dict.fromkeys(ch_names)
            for ch in ch_names:
                if feat_unique.__contains__(ch) == False:
                    feat_count[ch] = 0
                c = Counter(feat)
                feat_count.update(c)
                data = list(feat_count.values())
                channels = [str(k) for k in range(len(data))]

            # Plotting properties shared across topoplots and colorbar :
            kw_top = dict(margin=9 / 100, chan_offset=(0.0, 0.1, 0.0), chan_size=3)
            kw_cbar = dict(
                cbtxtsz=12,
                txtsz=10.0,
                width=0.3,
                txtcolor="white",
                cbtxtsh=1.8,
                rect=(0.0, -2.0, 1.0, 4.0),
                border=False,
            )
            sc = SceneObj(bgcolor="black", size=(600, 600))

            t_obj_1 = TopoObj(
                "topo",
                data,
                xyz=xy,
                line_color="white",
                line_width=7.0,
                chan_mark_color="lightgray",
                cmap=cmap,
                # line_color='#3498db',
                **kw_top,
            )
            cb_obj_1 = ColorbarObj(t_obj_1, cblabel="Counts", **kw_cbar)
            # Add the topoplot and the colorbar to the scene :
            sc.add_to_subplot(
                t_obj_1,
                row=0,
                col=0,
                title_color="white",
                width_max=400,
                title=f"DR features selection, {p}, {n} selected features",
            )
            sc.add_to_subplot(cb_obj_1, row=0, col=1, width_max=100)
            suffix2 = p + "_defaultBand_WithinSession"
            # screenshot doesn' work :'(
            save_pic_path = path_figures_root + "/Topo_Histogram_Chan_Select_" + suffix2
            save_as = os.path.join(save_pic_path, ".tiff")

            sc.preview(mpl=True)


def _plot_topo_features_clin(path_figures_root, dataset, df_res_chan):
    plt.style.use("dark_background")
    cmap = "magma"
    ch_pos = pd.read_csv(f"{os.getcwd()}/Datasets_montages_{dataset.code}.csv")
    ch_pos.rename(columns={ch_pos.columns[0]: "Label"})
    xy = ch_pos.to_numpy()[:, 1:3]
    xy = xy.astype(np.float)
    ch_names = ch_pos[ch_pos.columns[0]].to_list()
    pipelines = df_res_chan["pipeline"].unique()
    for p in pipelines:
        # create a dict that countains how many times a given channel is selected as a feature
        all_res_t = df_res_chan[
            df_res_chan["pipeline"] == p
        ]
        feat = all_res_t["chan_select"].explode().to_list()
        feat_unique = set(feat)
        feat_count = dict.fromkeys(ch_names)
        for ch in ch_names:
            if feat_unique.__contains__(ch) == False:
                feat_count[ch] = 0
            c = Counter(feat)
            feat_count.update(c)
            data = list(feat_count.values())
            channels = [str(k) for k in range(len(data))]

        # Plotting properties shared across topoplots and colorbar :
        kw_top = dict(margin=15 / 100, chan_offset=(0.0, 0.1, 0.0), chan_size=10)
        kw_cbar = dict(
            cbtxtsz=12,
            txtsz=10.0,
            width=0.3,
            txtcolor="white",
            cbtxtsh=1.8,
            rect=(0.0, -2.0, 1.0, 4.0),
            border=False,
        )
        sc = SceneObj(bgcolor="black", size=(600, 600))

        t_obj_1 = TopoObj(
            "topo",
            data,
            xyz=xy,  # levels=levels,
            # level_colors=level_colors,  # chan_mark_symbol='cross',
            line_color="white",
            line_width=7.0,
            chan_mark_color="lightgray",
            cmap=cmap,
            # line_color='#3498db',
            **kw_top,
        )
        cb_obj_1 = ColorbarObj(t_obj_1, cblabel="Counts", **kw_cbar)
        # Add the topoplot and the colorbar to the scene :
        sc.add_to_subplot(
            t_obj_1,
            row=0,
            col=0,
            title_color="white",
            width_max=300,
            title=f"DR features selection, {p}",
        )
        sc.add_to_subplot(cb_obj_1, row=0, col=1, width_max=100)
        suffix2 = p + "_defaultBand_clinical_PatientA"
        # screenshot doesn't work :'(
        save_pic_path = path_figures_root + "/Topo_Histogram_Chan_Select_" + suffix2
        sc.preview(mpl=True)

## dictionary that contains the colors associated to each case studied in the paper
dict_colors={"RegCSP+shLDA":"#0072B2","CSP+optSVM":'#E69F00','FgMDM':'#F0E442','cov':'#009E73',
            "ensemble-noDR_2":'#D55E00',
            "ensemble-noDR_best":'#F5E4C9',
            'ensemble-noDR_3':'#CC79A7',
            'ensemble-noDR_4':'#F79EB4',
            'ensemble-DR':"#85A6A3",
            "instantaneous":"#576d64","imcoh":"#c4d5a8",
            "plv":"#64405a","pli":"#9e7089","wpli2_debiased":"#ad96a9","aec":"#1E3F5A",
            "delta": "#f16745",
            "theta": "#ffc65d",
            "alpha": "#7bc8A4",
            "beta":  "#4cc3d9",
            "gamma": "#93648d",
            "defaultBand": "#404040",
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

## Dimension Reduction - Healthy subjects - Schirrmeister dataset
d = "Schirrmeister2017"

#%% merge the infos in a single dataframe
spect=["imcoh", "instantaneous"]
nb_nodes=[16, 32, 48, 64, 80, 96]#, 112]
ch_pos = pd.read_csv(f"{os.getcwd()}/Datasets_montages_Schirrmeister2017.csv")
ch_pos.rename(columns={ch_pos.columns[0]: "Label"})
xy = ch_pos.to_numpy()[:, 1:3]
xy = xy.astype(np.float)
ch_names = ch_pos[ch_pos.columns[0]].to_list()
DR_chan_select=pd.DataFrame()
columns=["subject", "pipeline", "score", "n_dr", "chan_select"]
val=[]
for s in range(14):
    for sp in spect:
        for nd in nb_nodes:
            filename=basedir+'/2_Dataset-npz/Schirrmeister2017/' + "opt-DR-ch_select-"+ str(s+1)+ "_"+sp+"-0.05_"+str(nd)
            with np.load(filename+'.npz', allow_pickle=True) as data:
                idx_chan = data['node_select']
                acc = data['acc']
                label_chan=[ch_names[index] for index in idx_chan]
                values=[s+1, sp, acc, nd, label_chan]
                zipped = zip(columns, values)
                a_dictionary = dict(zipped)
                val.append(a_dictionary)

DR_chan_select = DR_chan_select.append(val, True)
_plot_topo_features_extensive(path_figures_root=path_figures_root, dataset=Schirrmeister2017(), DR_chan_select=DR_chan_select)


palette_DR=sns.color_palette([dict_colors["instantaneous"],
                           dict_colors["imcoh"]])
#%% plot acc
pipelines = DR_chan_select["pipeline"].unique()
plt.style.use("dark_background")
results_plot = DR_chan_select

for s in range(14):
    results_plot_subj = results_plot[results_plot["subject"] == s+1]
    results_plot_subj["score"] = results_plot_subj.score.astype(float)
    g = sns.relplot(
        data=results_plot_subj,
        x="n_dr",
        y="score",
        hue="pipeline",
        kind="line",
        height=3,
        aspect=2,
        palette=palette_DR
    )
    leg = g._legend
    leg.set_bbox_to_anchor([1, 0.7])
    plt.title("Subject " + str(s+1))
    plt.ylim(0.4, 1)
    plt.savefig(
        path_figures_root
        + "/Schirrmeister2017/DR_chan_select/Opt_DR_IndivSubj_ImCoh_Instantaneous_paper_Schirrmeister2017_lines_indiv_subj"
        + str(s+1)
        + ".pdf",
        dpi=300,
    )

results_plot["score"] = results_plot.score.astype(float)
g = sns.relplot(
    data=results_plot,
    x="n_dr",
    y="score",
    hue="pipeline",
    kind="line",
    height=3,
    aspect=2,
    palette=palette_DR
)
leg = g._legend
leg.set_bbox_to_anchor([1, 0.7])
plt.suptitle("Group results")
plt.ylim(0.4, 1)
plt.savefig(
    path_figures_root
    + "/Schirrmeister2017/DR_chan_select/Opt_DR_AllSubj_ImCoh_Instantaneous_paper_Schirrmeister2017_lines_Group_10subj.pdf",
    dpi=300,
)


g = sns.relplot(
    data=results_plot,
    x="n_dr",
    y="score",
    hue="pipeline",
    kind="line",
    col="subject",
    col_wrap=5,
    height=3,
    aspect=2,
    palette=palette_DR
)
leg = g._legend
leg.set_bbox_to_anchor([1, 0.7])
plt.suptitle("Group results")
plt.ylim(0.4, 1)
plt.savefig(
    path_figures_root
    + "/Schirrmeister2017/DR_chan_select/Opt_DR_AllSubjIndiv_ImCoh_Instantaneous_paper_Schirrmeister2017_lines_Group_10subj.pdf",
    dpi=300,
)

## Dimension Reduction - Clinical application
d = "004-2015"
results_2class_rf_DRnoDR_optselect=pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf-invertloop-DRnoDR_IndivPreselectPaper.csv"
)
results_2class_rf_noDR = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf.csv"
)
results_2class_rf_noDR_preselect = pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf-invertloop-noDR_PreselectPaper.csv"
)
results_2class_rf_noDR_preselect["pipeline"] = results_2class_rf_noDR_preselect["pipeline"].replace(
   "ensemble-noDR","ensemble-noDR_preselectPaper"
)
# load DR results:
results_2class_rf_DR=pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf-invertloop-DR.csv"
)
# load DR results & preselect cf infos paper:
results_2class_rf_DR_preselect=pd.read_csv(
    path_csv_root + d + "/OptEnsemble-coreFC-allsubject-memory-2class-rf-invertloop-DR_PreselectPaper.csv"
)
results_2class_rf_DR_preselect["pipeline"] = results_2class_rf_DR_preselect["pipeline"].replace(
   "ensemble-DR","ensemble-DR_preselectPaper"
)
results_2class_rf=pd.concat((results_2class_rf_noDR,results_2class_rf_noDR_preselect,results_2class_rf_DR, results_2class_rf_DR_preselect, results_2class_rf_DRnoDR_optselect))
results_2class_rf["dataset"] = results_2class_rf["dataset"].replace(
    "004-2015", "004-2015-2classes_rf"
)
results_004_2015 = results_2class_rf

# to take into account results from Scherer et al' paper
results_004_2015["subject"] = results_004_2015["subject"].replace(1,"A")
results_004_2015["subject"] = results_004_2015["subject"].replace(2,"C")
results_004_2015["subject"] = results_004_2015["subject"].replace(3,"D")
results_004_2015["subject"] = results_004_2015["subject"].replace(4,"E")
results_004_2015["subject"] = results_004_2015["subject"].replace(5,"F")
results_004_2015["subject"] = results_004_2015["subject"].replace(6,"G")
results_004_2015["subject"] = results_004_2015["subject"].replace(7,"H")
results_004_2015["subject"] = results_004_2015["subject"].replace(8,"J")
results_004_2015["subject"] = results_004_2015["subject"].replace(9,"L")


## focus on patients  A (lock-in syndrome)
subgroup_list=["A"]
palette_DR=sns.color_palette([dict_colors["RegCSP+shLDA"],
                           dict_colors["CSP+optSVM"],
                           dict_colors["FgMDM"],
                           dict_colors["cov"],
                           dict_colors["ensemble-noDR_best"], # FUCONE
                            dict_colors["ensemble-DR"]    # FUCONE+DR
                                ])
results_004_2015_subgroup=results_004_2015[results_004_2015["subject"].isin(subgroup_list)]
list_ppl = [
    "RegCSP+shLDA",
    "CSP+optSVM",
    "FgMDM",
    "cov+elasticnet",
    "ensemble-noDR",
    "ensemble-DR",
]
hue_order_004_2015_sub_ppl = [
    "RegCSP+shLDA",
    "CSP+optSVM",
    "FgMDM",
    "cov+elasticnet",
    "ensemble-noDR",
    "ensemble-DR",
]
DR_tickslabels=["RegCSP+shLDA",
    "CSP+optSVM",
    "FgMDM",
    "Cov+EN",
    "FUCONE",
    "FUCONE-DR",]
g=sns.catplot(data=results_004_2015_subgroup[results_004_2015_subgroup["pipeline"].isin(list_ppl)], x="pipeline", y="score", #row="subject",
              kind="bar",
              order=hue_order_004_2015_sub_ppl,
              palette=palette_DR, height=4, aspect=3)
plt.ylim((0, 1))
plt.xticks(range(len(list_ppl)),
           DR_tickslabels)
plt.yticks(fontsize=18)
plt.xticks(fontsize=15)
plt.xlabel("Pipeline", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.suptitle("Patient A")
plt.savefig(
    path_figures_root
    + "Clinical_Opt_DR_BarPlot_PatientA.pdf",
    dpi=300,
)



## patient A - plot topos
from eeg_positions import get_elec_coords
spect=["imcoh", "instantaneous"]
list_ch_names=['F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']

coords = get_elec_coords(
    system="1010",
    dim="2d",
)
ch_pos_temp=coords.T.transpose()
ch_pos=ch_pos_temp[ch_pos_temp["label"].isin(list_ch_names)]
ch_2csv=ch_pos.set_index('label')
ch_2csv.to_csv("Datasets_montages_004-2015.csv")
xy = ch_pos.to_numpy()[:, 1:3]
xy = xy.astype(np.float)
ch_names = ch_pos[ch_pos.columns[0]].to_list()
DR_chan_select=pd.DataFrame()
columns=["subject", "pipeline", "score", "n_dr", "chan_select"]
val=[]
for sp in results_2class_rf_DR["subject"].unique()[:1]:
    for fc_met in spect:
        filename=basedir+'/2_Dataset-npz/004-2015/' + "opt-DR-ch_select-"+str(sp)+"_"+fc_met+"-0.05"
        with np.load(filename+'.npz', allow_pickle=True) as data:
            idx_chan = data['node_select']
            acc = data['acc']
            label_chan=[ch_names[index] for index in idx_chan]
            values=[sp, fc_met, acc, len(idx_chan), label_chan]
            zipped = zip(columns, values)
            a_dictionary = dict(zipped)
            val.append(a_dictionary)

DR_chan_select = DR_chan_select.append(val, True)
_plot_topo_features_clin(path_figures_root=path_figures_root, dataset=BNCI2015004(), df_res_chan=DR_chan_select)

##


