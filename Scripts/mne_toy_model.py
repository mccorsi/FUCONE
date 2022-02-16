import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial

import mne
from mne import make_ad_hoc_cov
from mne.datasets import sample
from mne.simulation import (simulate_sparse_stc, simulate_raw,
                            add_noise)
from pyriemann.estimation import Coherences, Covariances
from pyriemann.utils.distance import distance
#from Scripts.fc_pipeline import nearestPD
import seaborn as sns


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

def data_sin_src(times, amp=1.0, freq=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * freq * times + phase)


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference(projection=True)
raw = raw.pick_types(meg=False, eeg=True, exclude=['EEG 053'])
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude=['EEG 053'])

n_dipoles = 1  # number of dipoles to create
epoch_duration = 2.5  # duration of each epoch/event
n = 0  # harmonic number
rng = np.random.RandomState(42)  # random state (make reproducible)
sfreq = raw.info['sfreq']
times = np.arange(0, epoch_duration, 1 / sfreq)

dfun = partial(data_sin_src, amp=1e-7, freq=10.0, phase=0.0)
src = fwd['src']
stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                          data_fun=dfun, random_state=rng)

# fig, ax = plt.subplots(1)
# ax.plot(times, stc.data.T)
# plt.show()

raw_sim = simulate_raw(raw.info, [stc] * 1, forward=fwd, verbose=True)
cov = make_ad_hoc_cov(raw_sim.info)
add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
raw_sim.plot()

Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

eeg = raw_sim.get_data()
cov = Cov.fit_transform(eeg[np.newaxis, ...])[0]
coh = nearestPD(Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)[0])
imcoh = nearestPD(preproc(ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1))[0])

distance(cov, coh, metric="riemann")
distance(cov, imcoh, metric="riemann")


###############################################################################
# Amplitude

A = 1e-8
f = 10.0
ph = 0.0

Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
a_space = np.linspace(0.1, 4.0, 40) * A
for amp in a_space:
    dfun = partial(data_sin_src, amp=amp, freq=f, phase=ph)
    src = fwd['src']
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=dfun, random_state=rng, location='center')
    raw_sim = simulate_raw(raw.info, [stc] * 1, forward=fwd)
    cov = make_ad_hoc_cov(raw_sim.info)
    add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
    eeg = raw_sim.get_data()

    cov_mats.append(Cov.fit_transform(eeg[np.newaxis, ...])[0])
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

# cov_d = np.array(cov_d)
# coh_d = np.array(coh_d)
# imcoh_d = np.array(imcoh_d)

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
ax.set_xlim(0, 4 * A)
ax.xaxis.set_ticks(np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * A)
ax.xaxis.set_ticklabels(["0", "1", "2", "3", "4"])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(visible=False)
ax.legend(bbox_to_anchor=(1.14, 1), frameon=False, prop={'size': 15})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# plt.savefig(path_figures_root + "Toy_model_Synthetic_Amplitude.pdf", dpi=300)
plt.show()

###############################################################################
# Freq

A = 1e-4
f = 10.0
ph = 0.0

Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
f_space = np.linspace(4.0, 50.0, 47)
for freq in f_space:
    dfun = partial(data_sin_src, amp=A, freq=freq, phase=ph)
    src = fwd['src']
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=dfun, random_state=rng, location='center')
    raw_sim = simulate_raw(raw.info, [stc] * 1, forward=fwd)
    cov = make_ad_hoc_cov(raw_sim.info)
    add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
    eeg = raw_sim.get_data()
    
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
ax.legend() # bbox_to_anchor=(1.14, 1), frameon=False, prop={'size': 15})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# plt.savefig(path_figures_root + "Toy_model_Synthetic_Frequency.pdf", dpi=300)
plt.show()


###############################################################################
# Phase

A = 1e-5 # 1
f = 10.0
ph = 0.0

Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov_mats, coh_mats, imcoh_mats = [], [], []
p_space = np.linspace(0, 6 * np.pi, 60)
for dec_phase in p_space:
    dfun = partial(data_sin_src, amp=A, freq=f, phase=dec_phase)
    src = fwd['src']
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=dfun, random_state=rng)
    raw_sim = simulate_raw(raw.info, [stc] * 1, forward=fwd)
    cov = make_ad_hoc_cov(raw_sim.info)
    add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
    eeg = raw_sim.get_data()

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
# plt.savefig(path_figures_root + "Toy_model_Synthetic_Phase.pdf", dpi=300)
plt.show()

