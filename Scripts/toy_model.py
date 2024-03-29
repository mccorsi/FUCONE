##
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from pyriemann.estimation import Coherences, Covariances
from pyriemann.utils.distance import distance
#from Scripts.fc_pipeline import nearestPD
import seaborn as sns


sfreq = 250
duration = 5 # in seconds

def sin_source(amp=1.0, freq=1.0, phase=0.0):
    t = np.arange(0, duration, 1 / sfreq)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def preproc(S):
    c = np.triu(S[0])
    n = c.shape[0]
    c = c + c.T - np.diag(np.diag(c)) + np.identity(n)
    return c[np.newaxis, ...]


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via eigenvalues"""
    if np.any(np.linalg.eigvals(B) < 0.0):
        return False
    else:
        return True


def nearestPD(A, reg=1e-6):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
        print(f"Correction: {-mineig * k ** 2 + spacing:.2}")

    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3

## initialize signals infos & plot
if os.path.basename(os.getcwd()) == "FUCONE":
    os.chdir("Database")
basedir = os.getcwd()
path_figures_root = basedir + "/0_Figures/Toy_Model/"

A1, A2 = 2, 0.5
f1, f2 = 8.0, 12.0
ph1, ph2 = 0, np.pi / 2
noise = 0.2
n_sources = 2
n_elec = 4

s1 = sin_source(A1, f1, ph1)
s2 = sin_source(A2, f2, ph2)
S = np.stack([s1, s2])
mix = np.random.randn(n_sources, n_elec)

eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
eeg = eeg.T
time = np.arange(len(eeg[0]))/sfreq

plt.close("all")
plt.style.use("dark_background")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, 9))
for i in range(n_elec):
    plt.plot(time, eeg.T[:,i], color=colors[i], linewidth=6)
#plt.plot(time,eeg.T, linewidth=6)
plt.legend(('elec. #1', 'elec. #2', 'elec. #3', 'elec. #4'),
           loc='best', frameon=False, prop={'size': 15})
ax.set_facecolor("black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylim([-6, 6])
plt.xlim([0, 1.2])
plt.grid(False)
plt.xlabel('Time (s)', fontsize=30)
plt.ylabel('Amplitude (A.U.)', fontsize=30)
plt.title('Simulated signals from 4 synthetic electrodes', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(path_figures_root + "Toy_model_Synthetic_Signals.pdf", dpi=300)


## Compute covariances & FC estimators
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov = Cov.fit_transform(eeg[np.newaxis, ...])[0]
coh = nearestPD(Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)[0])
imcoh = nearestPD(preproc(ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1))[0])

distance(cov, coh, metric="riemann")
distance(cov, imcoh, metric="riemann")

plt.close('all')
ax = sns.heatmap(cov, linewidth=0.5, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)
plt.xticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.title('Covariance - reference')
plt.savefig(path_figures_root + "Toy_model_Synthetic_Cov.pdf", dpi=300)
plt.show()

plt.close('all')
ax = sns.heatmap(coh, linewidth=0.5, cmap="YlGnBu_r", xticklabels=False, yticklabels=False,vmax=.3, square=True)
plt.xticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.title('Coherence - reference')
plt.savefig(path_figures_root + "Toy_model_Synthetic_Coh.pdf", dpi=300)
plt.show()

plt.close('all')
ax = sns.heatmap(imcoh, linewidth=0.5, cmap="YlGnBu_r", xticklabels=False, yticklabels=False,vmax=.1, square=True)
plt.xticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['#1', '#2', '#3', '#4'])
plt.title('ImCoh - reference')
plt.savefig(path_figures_root + "Toy_model_Synthetic_ImCoh.pdf", dpi=300)
plt.show()


###############################################################################
# Phase

A = 1
f = 10.0
ph = 0.0
noise = 0.1
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
p_space = np.linspace(0, 6 * np.pi, 60)
for dec_phase in p_space:
    s1 = sin_source(A, f, ph)
    s2 = sin_source(A, f, dec_phase)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov_mats.append(Cov.fit_transform(eeg[np.newaxis, ...])[0])
    coh_mats.append(nearestPD(Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)[0]))
    imcoh_mats.append(
        nearestPD(preproc(ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1))[0])
    )

cov_d, coh_d, imcoh_d = [], [], []
idref = len(p_space) // 2
for i in range(len(p_space)):
    cov_d.append(distance(cov_mats[i], cov_mats[idref]))
    coh_d.append(distance(coh_mats[i], coh_mats[idref]))
    imcoh_d.append(distance(imcoh_mats[i], imcoh_mats[idref]))

cov_d = np.array(cov_d)
coh_d = np.array(coh_d)
imcoh_d = np.array(imcoh_d)

mpl.style.use("seaborn-muted")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
lcov = ax.plot(
    p_space,
    cov_d,
    label=r"$\delta(\mathrm{cov}_{\mathrm{ref}}, \mathrm{cov})$",
    color="C0", linewidth=6
)
ax2 = ax.twinx()
lcoh = ax2.plot(
    p_space,
    coh_d,
    label=r"$\delta(\mathrm{coh}_{\mathrm{ref}}, \mathrm{coh})$",
    color="C1", linewidth=6
)
limcoh = ax2.plot(
    p_space,
    imcoh_d,
    label=r"$\delta(\mathrm{imcoh}_{\mathrm{ref}}, \mathrm{imcoh})$",
    color="C2", linewidth=6
)
ax.set_xlabel("Source phase", fontsize=30)
# ax.xaxis.set_ticks([1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
# ax.xaxis.set_ticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
ax.xaxis.set_ticks([1 * np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi, 5 * np.pi])
ax.xaxis.set_ticklabels([r"$-3\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"], fontsize=24)
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(20))
ax.grid(visible=False, axis="both", which="both")
ax2.grid(visible=False)
ax.tick_params(axis='y', labelsize=24)
ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
#ax.yaxis.set_ticklabels([])
#ax2.yaxis.set_ticks([])
ax.set_ylabel("$\delta$", fontsize=30)

plt.yticks(fontsize=24)
ax.set_ylabel("$\delta$ (cov)", fontsize=30)
ax2.set_ylabel("$\delta$ (coh, imcoh)", fontsize=30)

ax.set_xlim(1.5 * np.pi, 4.5 * np.pi)
ax.set_title("Phase influence", fontsize=30)
lns = lcov + lcoh + limcoh
labs = [l.get_label() for l in lns]
#ax.legend(lns, labs, bbox_to_anchor=(1.00, 1.114), frameon=False, prop={'size': 21})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(path_figures_root + "Toy_model_Synthetic_Phase.pdf", dpi=300)
plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.plot(p_space, imcoh_d, label=r"$\delta(\mathrm{imcoh}_{\mathrm{ref}}, \mathrm{imcoh})$")
# ax.set_xlim(2 * np.pi, 4 * np.pi)
# ax.set_xlabel("source phase")
# ax.xaxis.set_ticks([2 * np.pi, 3 * np.pi, 4 * np.pi])
# ax.xaxis.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
# ax.set_title("Variation of the source phase")
# ax.legend(loc="upper right")


###############################################################################
# Amplitude

A = 1
f1, f2 = 10.0, 14.0
ph1, ph2 = 0.0, 0.75 * np.pi
noise = 0.0
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
a_space = np.linspace(0.1, 4.0, 40)
for amp in a_space:
    s1 = sin_source(A, f1, ph1)
    s2 = sin_source(amp, f2, ph2)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov_mats.append(nearestPD(Cov.fit_transform(eeg[np.newaxis, ...])[0]))
    coh_mats.append(nearestPD(Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)[0]))
    imcoh_mats.append(
        nearestPD(preproc(ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1))[0])
    )

cov_d, coh_d, imcoh_d = [], [], []
idref = 9
for i in range(len(a_space)):
    cov_d.append(distance(cov_mats[i], cov_mats[idref]))
    coh_d.append(distance(coh_mats[i], coh_mats[idref]))
    imcoh_d.append(distance(imcoh_mats[i], imcoh_mats[idref]))

mpl.style.use("seaborn-muted")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(
    a_space,
    cov_d,
    label=r"$\delta(\mathrm{cov}_{\mathrm{ref}}, \mathrm{cov})$",
    color="C0", linewidth=6
)
ax.plot(
    a_space,
    coh_d,
    label=r"$\delta(\mathrm{coh}_{\mathrm{ref}}, \mathrm{coh})$",
    color="C1", linewidth=6
)
ax.plot(
    a_space,
    imcoh_d,
    label=r"$\delta(\mathrm{imcoh}_{\mathrm{ref}}, \mathrm{imcoh})$",
    color="C2", linewidth=6
)
# ax.vlines(1.., ymin=-1.1, ymax=1.0, linestyles="dashed", color="k")
ax.set_xlabel("Amplitude ratio", fontsize=30)
ax.set_title("Amplitude influence", fontsize=30)
ax.set_ylabel(r"$\delta$", fontsize=30)
#ax.yaxis.set_ticklabels([])
ax.set_xlim(0, 4)
ax.xaxis.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.xaxis.set_ticklabels(["0", "1", "2", "3", "4"])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(visible=False)
ax.legend(bbox_to_anchor=(1.14, 1), frameon=False, prop={'size': 15})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(path_figures_root + "Toy_model_Synthetic_Amplitude.pdf", dpi=300)
plt.show()


###############################################################################
# freq

A = 1
f = 10.0
ph = 0.0
noise = 0.0
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
f_space = np.linspace(4.0, 50.0, 47)
for freq in f_space:
    s1 = sin_source(A, f, ph)
    s2 = sin_source(A, freq, ph)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov_mats.append(nearestPD(Cov.fit_transform(eeg[np.newaxis, ...])[0]))
    coh_mats.append(nearestPD(Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)[0]))
    imcoh_mats.append(
        nearestPD(preproc(ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1))[0])
    )

cov_d, coh_d, imcoh_d = [], [], []
idref = 6
for i in range(len(f_space)):
    cov_d.append(distance(cov_mats[i], cov_mats[idref]))
    coh_d.append(distance(coh_mats[i], coh_mats[idref]))
    imcoh_d.append(distance(imcoh_mats[i], imcoh_mats[idref]))

mpl.style.use("seaborn-muted")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(
    f_space,
    cov_d,
    label=r"$\delta(\mathrm{cov}_{\mathrm{ref}}, \mathrm{cov})$",
    color="C0", linewidth=6
)
ax.plot(
    f_space,
    coh_d,
    label=r"$\delta(\mathrm{coh}_{\mathrm{ref}}, \mathrm{coh})$",
    color="C1", linewidth=6
)
ax.plot(
    f_space,
    imcoh_d,
    label=r"$\delta(\mathrm{imcoh}_{\mathrm{ref}}, \mathrm{imcoh})$",
    color="C2", linewidth=6
)
# ax.vlines(10., ymin=-1.1, ymax=1.0, linestyles="dashed", color="k")
# ax.set_ylim(-1.08, 0.9)
#ax.yaxis.set_ticklabels([])
ax.set_ylabel(r"$\delta$", fontsize=30)
ax.xaxis.set_ticks([10, 20, 30, 40, 50])
ax.xaxis.set_ticklabels([r"$f$", r"$2f$", r"$3f$", r"$4f$", r"$5f$"])
ax.set_xlabel("Frequency ratio", fontsize=30)
ax.set_title("Frequency influence", fontsize=30)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(visible=False)
ax.legend(bbox_to_anchor=(1.14, 1), frameon=False, prop={'size': 15})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(path_figures_root + "Toy_model_Synthetic_Frequency.pdf", dpi=300)
plt.show()

##

