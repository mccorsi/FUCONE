"""
==============================================================
SimEEG - toy model - FUCONE
===============================================================
T

"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pandas as pd
from scipy.io import loadmat
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline

from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import MDM, FgMDM, TSclassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from mne.decoding import CSP
from tqdm.notebook import trange, tqdm

import os
##
if os.path.basename(os.getcwd()) == "moabb_connect":
    os.chdir("Database")
basedir = os.getcwd()

freqbands = {
    "defaultBand": [8, 35],
}
spectral_met = [
    "coh",
    "imcoh",
    "cov"
]

path_data_root = "/moabb_connect/100_Matlab_scripts/"
os.chdir(os.path.join(basedir, path_data_root))


l_freq, h_freq = 8, 30
start, stop = 0.2, 4
scoring = 'accuracy'

##
param_svc = {'C': np.logspace(-2, 2, 10)}
gssvc = GridSearchCV(SVC(kernel='linear'), param_svc, cv=3)
param_elasticnet = {'1_ratio': 1 - np.logspace(-2, 0, 10),
                    'cv': 3}
elasticnetcv = ElasticNetCV(*param_elasticnet)

pipelines = {}
pipelines['MDM'] = make_pipeline(
    Covariances(estimator='scm'),
    MDM(metric='riemann', n_jobs=-1))
pipelines['fgMDM'] = make_pipeline(
    Covariances(estimator='scm'),
    FgMDM(metric='riemann', tsupdate=False, n_jobs=-1))
pipelines['TS_SVM'] = make_pipeline(
    Covariances(estimator='scm'),
    TangentSpace(metric='riemann'),
    gssvc)

pipelines['CSP_LDA'] = make_pipeline(
    CSP(n_components=6),
    LDA())



##

d = loadmat(basedir+"/moabb_connect/100_Matlab_scripts/NoiseEffect_V2.mat", squeeze_me=True)
list_noise=d['simu_noise'][0,:]
for kk_noise, ampl_noise in enumerate(list_noise):
    data=d['simu_noise'][1,kk_noise]
    labels=d['simu_noise'][2,kk_noise]
    Covs = Covariances(estimator='scm').fit_transform(data)
    #data.shape

    sfreq = 250
    channel_names = ['Fp1','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','P9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','P10','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO9','PO7','PO3','POz','PO4','PO8','PO10','O1','Oz','O2']

    channel_types = 64 * ["eeg"]
    montage = mne.channels.make_standard_montage('standard_1010')
    info = mne.create_info(channel_names, sfreq, channel_types, montage)
    tmin = 0.2
    event_id = {'rest': 1, 'imagery': 2}
    events = np.array([[t * 10 * sfreq, 0, l] for t, l in enumerate(labels)])
    epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    epochs.set_montage(montage)


    X = epochs.get_data()
    y = epochs.events[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    for name in tqdm(pipelines):
        acc = cross_val_score(pipelines[name], X, y, cv=cv, scoring=scoring)
        res = {'subject': str(s),
               'score': acc.mean(),
               'pipeline': name}
        all_res.append(res)
    df_std = pd.DataFrame(all_res)


##
