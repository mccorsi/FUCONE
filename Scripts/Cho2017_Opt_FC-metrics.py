"""
==============================================================
Cho2017 - Parameters optimization: FC metrics - FUCONE
===============================================================
This module is design to select the FC metrics that enhance the accuracy

"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from moabb.paradigms import LeftRightImagery
from moabb.datasets import Cho2017

from fc_pipeline import FunctionalTransformer, EnsureSPD, WithinSessionEvaluationFCDR

##
if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
basedir = os.getcwd()

threshold = [0.05, 0.01, 0.005]
nb_nodes = [5, 10, 15]
datasets = [Cho2017()]

freqbands = {
    #     "delta": [2, 4],
    #     "theta": [4, 8],
    #     "alpha": [8, 12],
    #     "beta": [15, 30],
    #     "gamma": [30, 45],
    "defaultBand": [8, 35],
}
spectral_met = [
    "coh",
    "imcoh",
    "plv",
    "pli",
    "pli2_unbiased",
    "wpli",
    "wpli2_debiased",
    "aec",
]

# list of pre-selected subjects 
subj = [14, 43, 50, 35, 3, 29, 7, 17, 40, 38]
print(
    "#################" + "\n"
    "List of pre-selected subjects from Cho2017 (5 best and 5 least performant): "
    + "\n"
    + str(subj)
    + "\n"
    "#################"
)

for d in datasets:
    d.subject_list = subj
    d.n_sessions = 1
    path_csv_root = "1_Dataset-csv/" + d.code
    path_data_root = "2_Dataset-npz/" + d.code
    path_data_root_chan = path_data_root + "/Chan_select/"
    path_figures_root = "0_Figures/" + d.code

    os.chdir(os.path.join(basedir, path_data_root))

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
        results.to_csv(
            basedir
            + path_csv_root
            + "/res_np_single_pipelines_Cho2017_preSelectedSubj_OptFCMetrics_"
            + f
            + ".csv"
        )