##
from moabb.datasets import (
    BNCI2014001 ,
    BNCI2015001 ,
    BNCI2015004 ,
    Zhou2016
)
from moabb.analysis.meta_analysis import (
    find_significant_differences ,
)

import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd
import numpy as np
import seaborn as sns
import os


# %%
if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
if os.path.basename(os.getcwd()) == "Scripts":
    os.chdir("../Database")
basedir = os.getcwd()

# %%
datasets = [ BNCI2014001 () , BNCI2015001 () , BNCI2015004 () , Zhou2016 () ]
os.chdir ( basedir )

for d in datasets:
    if d.code == "Zhou 2016":
        d.code = "Zhou-2016"
    path_csv_root = basedir + "/1_Dataset-csv/" + d.code + "/"
    path_data_root = basedir + "/2_Dataset-npz/" + d.code + "/"
    path_figures_root = basedir + "/0_Figures/" + d.code + "/"

    res = pd.read_csv (
        path_csv_root
        + "diversity-2class-rf.csv"
    )
    diversity = res[ "diversity" ]
    fpfn = res[ 'FP+FN' ]
    relative_diversity = diversity / fpfn
    nb_trials = res[ 'nb_trials' ]
    res[ 'rel_diversity' ] = relative_diversity
    potential_improv = diversity / nb_trials
    res[ 'pot_improv' ] = potential_improv

    res.to_csv (
        path_csv_root
        + "diversity-metrics-2class-rf.csv"
    )

    list_feat = [ 'instantaneous' , 'imcoh' , 'plv' , 'pli' , 'wpli2_debiased' , 'aec' ]
    dict_colors = {"RegCSP+shLDA": "#0072B2" , "CSP+optSVM": '#B89ED5' , 'FgMDM': '#314a93' ,
                   'cov': '#009E73' ,
                   "ensemble-noDR_2": '#1389E6' ,
                   "ensemble-noDR_best": "#BE4E4E" ,
                   'ensemble-noDR_3': '#F97738' ,
                   'ensemble-noDR_4': '#ffd02d' ,
                   "instantaneous": "#576d64" , "imcoh": "#c4d5a8" ,
                   "plv": "#64405a" , "pli": "#9e7089" , "wpli2_debiased": "#ad96a9" , "aec": "#3C648E" ,
                   "delta": "#f16745" ,
                   "theta": "#ffc65d" ,
                   "alpha": "#7bc8A4" ,
                   "beta": "#4cc3d9" ,
                   "gamma": "#93648d" ,
                   "defaultBand": "#F98790" ,
                   }
    colors_fc_paper = [
        dict_colors[ "instantaneous" ] ,
        dict_colors[ "imcoh" ] ,
        dict_colors[ "plv" ] ,
        dict_colors[ "pli" ] ,
        dict_colors[ "wpli2_debiased" ] ,
        dict_colors[ "aec" ]
    ]

    plt.style.use ( "dark_background" )
    sns.catplot ( data=res[ res[ "feature" ].isin ( list_feat ) ] ,
                  x="feature" ,
                  y="diversity" ,
                  kind="box" ,
                  height=3 ,
                  aspect=3 ,
                  order=list_feat ,
                  palette=sns.color_palette ( colors_fc_paper ) )
    plt.title ( d.code )
    plt.savefig (
        path_figures_root + "/Diversity_" + d.code + "_2class-rf.pdf" , dpi=300
    )
    sns.catplot ( data=res[ res[ "feature" ].isin ( list_feat ) ] ,
                  x="feature" ,
                  y="pot_improv" ,
                  kind="box" ,
                  height=3 ,
                  aspect=3 ,
                  order=list_feat ,
                  palette=sns.color_palette ( colors_fc_paper ) )
    plt.title ( d.code )
    plt.savefig (
        path_figures_root + "/PotImprov_" + d.code + "_2class-rf.pdf" , dpi=300
    )
    sns.catplot ( data=res[ res[ "feature" ].isin ( list_feat ) ] ,
                  x="feature" ,
                  y="rel_diversity" ,
                  kind="box" ,
                  height=3 ,
                  aspect=3 ,
                  order=list_feat ,
                  palette=sns.color_palette ( colors_fc_paper ) )
    plt.title ( d.code )
    plt.savefig (
        path_figures_root + "/RelDiversity_" + d.code + "_2class-rf.pdf" , dpi=300
    )
