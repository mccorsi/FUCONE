"""
==============================================================
Cho2017 - Parameters optimization: Frequency band - Rigoletto
===============================================================
This module is design to select the frequency bands that enhance the accuracy

"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import os.path as osp
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression,
)

from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Coherences

from moabb.paradigms import LeftRightImagery
from moabb.datasets import (
    Cho2017,
)

from fc_pipeline import (
    FunctionalTransformer,
    EnsureSPD,
    WithinSessionEvaluationFCDR,
    AvgFC,
)

# %%
if os.path.basename(os.getcwd())=='RIGOLETTO':
    os.chdir('moabb_connect')
basedir = os.getcwd()

#%%
threshold = [0.05, 0.01, 0.005]
nb_nodes = [5, 10, 15]
datasets = [Cho2017()]
# TODO: put here the list of pre-selected subjects - done
subj = [14, 43, 50, 35, 3, 29, 7, 17, 40, 38]
print( "#################" + "\n"
        "List of pre-selected subjects from Cho2017 (5 best and 5 least performant): " + "\n"
       + str(subj) + "\n" 
       "#################")
# TODO: put here the list of pre-selected FC metrics
spectral_met = ["coh", "imcoh", "plv", "wpli2_debiased", 'instantaneous', 'lagged'] #TODO: update
print( "#################" + "\n"
        "List of pre-selected FC metrics: " + "\n"
       + str(spectral_met) + "\n" 
       "#################")

freqbands = {
        "delta": [2, 4],
        "theta": [4, 8],
        "alpha": [8, 12],
        "beta": [15, 30],
        "gamma": [30, 45],
    "defaultBand": [8, 35],
}

for d in datasets:
    d.subject_list = subj
    d.n_sessions = 1

    path_csv_root = basedir + "/1_Dataset-csv/" + d.code
    path_data_root = basedir + "/2_Dataset-npz/" + d.code
    path_data_root_chan = path_data_root + "/Chan_select/"
    path_figures_root = basedir + "/0_Figures/" + d.code

    os.chdir(path_data_root)

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]

        paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)

        pipelines = {}
        for sm in spectral_met:
            ft = FunctionalTransformer(
                delta=1, ratio=0.5, method=sm, fmin=fmin, fmax=fmax
            )

            pname_preDR = sm + "+elasticnet"
            pipelines[pname_preDR] = Pipeline(
                steps=[
                    ("sm", ft),
                    ("spd", EnsureSPD()),
                    ("tg", TangentSpace(metric="riemann")),
                    (
                        "LogistReg",
                        LogisticRegression(
                            penalty="elasticnet",
                            l1_ratio=0.15,
                            intercept_scaling=1000.0,
                            solver="saga",
                        ),
                    ),
                ]
            )

        evaluation = WithinSessionEvaluationFCDR(
            fmin=fmin,
            fmax=fmax,
            paradigm=paradigm,
            datasets=[d],
            n_jobs=-1,
            random_state=42,
            return_epochs=True,
            overwrite=True,
        )
        results = evaluation.process(pipelines)
        # Freq = pd.DataFrame([f]*len(results), columns={'FreqBand'})
        # results=pd.concat((results,Freq))
        results['FreqBand'] = f
        results.to_csv(
                path_csv_root
                + "/res_np_single_pipelines_Cho2017_preSelectedSubj_OptFreqBands_"
                + f
                + ".csv"
            )

# %% script to compare results between datasets & plots
import seaborn as sns
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)
from moabb.analysis.plotting import summary_plot
import matplotlib.pyplot as plt

# path_figures_root = "0_Figures/" + d.code

plt.style.use("dark_background")

paradigm = LeftRightImagery(fmin=8, fmax=35)
ch_labels = dict()  # dict that contains all the channel labels
all_res_temp = pd.DataFrame()


for d in datasets:
    path_csv_root = basedir + "/1_Dataset-csv/" + d.code
    for f in freqbands:
        res = pd.read_csv(
                path_csv_root
                + "/res_np_single_pipelines_Cho2017_preSelectedSubj_OptFreqBands_"
                + f
                + ".csv"
            )
        all_res_temp = pd.concat([all_res_temp, res])

all_res_temp.to_csv(
                path_csv_root
                + "/res_np_single_pipelines_Cho2017_preSelectedSubj_AllFreq_OptFreqBands_"
                + f
                + ".csv"
            )

results=pd.DataFrame()
for f in freqbands:
    results_f_temp=pd.read_csv("1_Dataset-csv/"+d.code+"/res_np_single_pipelines_"+d.code+"_preSelectedSubj_OptFreqBands_"+f+".csv")
    results_f=results_f_temp.head(n=60)
    results_f=results_f.drop(columns=results_f.keys()[0])
    results_f["FreqBand"]=[f]*len(results_f)
    results=pd.concat((results,results_f))
results.to_csv("1_Dataset-csv/"+d.code+"/res_np_single_pipelines_"+d.code+"_preSelectedSubj_AllFreq_OptFreqBands_defaultBand.csv")
