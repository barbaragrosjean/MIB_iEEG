import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json
import sys
sys.path.append("..")
from src.setting import GetInfo, PROJECT_PATH

from sklearn.decomposition import PCA

ieeg_datapath = 'ieeg_shortWOBS_fs250'
meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
meg_outpath ='MEG/dataMEG'

os.path.isdir(meg_datapath)
meg_subj_list = [file.replace('_sensor.p', '') for file in os.listdir(meg_outpath) if '_sensor' in file]
print('MEG Subject number : ', len(meg_subj_list))

ieeg_subj_list = [file.replace('_epochs.p', '') for file in os.listdir(ieeg_datapath) if file[-len('epochs.p'):] == 'epochs.p']
print('iEEG Subject number : ', len(ieeg_subj_list))

info_file = ieeg_datapath + f'/{ieeg_subj_list[0]}_info.json'
with open(info_file) as json_data:
    d = json.load(json_data)
    time_ieeg = d['time_epoch']

project_path = '../' + PROJECT_PATH
coord, areas, elect_list, subj_list, regions_ieeg = GetInfo(ieeg_subj_list, data_path=ieeg_datapath, project_path=project_path)
coord=np.array(coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = coord/1000

dt = np.mean(np.diff(time_ieeg))
fs = 1 / dt
f_min = 0.5
f_max = 100
n_frequencies = 86
frequencies = np.geomspace(f_min,f_max,n_frequencies)
bandwidth = [max(0.5,0.20 * freq) for freq in frequencies]

out_path = 'out/data'
with open(out_path + '/ieeg_data_mean.p', "rb") as f:
    ieeg_data = pickle.load(f)

with open(out_path + '/data_matched.p', "rb") as f:
    data_matched = pickle.load(f)  

data_matched = data_matched.mean(0).T
ieeg_data = ieeg_data.mean(0).T

def gaussian_frequency_filter(X, fs, center_frequency, bandwidth):
    """
    Frequency-domain Gaussian filtering.

    Parameters
    ----------
    X : array, shape (n_samples, n_channels)
        Broadband multivariate signal.

    fs : float
        Sampling frequency in Hz.

    center_frequency : float
        Center frequency of the Gaussian filter.

    bandwidth : float
        Full width at half maximum (FWHM) in Hz.

    Returns
    -------
    X_filtered : array, shape (n_samples, n_channels)
        Narrowband filtered signal.
    """

    X = np.asarray(X, dtype=float)

    n_samples = X.shape[0]

    # ---------------------------------------------------------
    # Frequency axis
    # ---------------------------------------------------------

    frequencies = np.fft.fftfreq(
        n_samples,
        d=1 / fs
    )

    # ---------------------------------------------------------
    # Convert FWHM to sigma
    # ---------------------------------------------------------

    sigma = bandwidth / (
        2 * np.sqrt(2 * np.log(2))
    )

    # ---------------------------------------------------------
    # Gaussian frequency-domain kernel
    # ---------------------------------------------------------

    kernel = np.exp(
        -0.5 * (
            (np.abs(frequencies) - center_frequency)
            / sigma
        )**2
    )

    # ---------------------------------------------------------
    # FFT
    # ---------------------------------------------------------

    X_fft = np.fft.fft(
        X,
        axis=0
    )

    # ---------------------------------------------------------
    # Apply Gaussian filter
    # ---------------------------------------------------------

    X_fft_filtered = (
        X_fft
        * kernel[:, None]
    )

    # ---------------------------------------------------------
    # Inverse FFT
    # ---------------------------------------------------------

    X_filtered = np.fft.ifft(
        X_fft_filtered,
        axis=0
    ).real

    return X_filtered

X_centered = ieeg_data #- np.mean(ieeg_data, axis=0)
eigenvalue_spectrum = np.zeros((len(frequencies), X_centered.shape[0]))
explained_variance_ratio = np.zeros((len(frequencies), X_centered.shape[0]))

for i, frequency in enumerate(frequencies):
    print(frequency)
    bandwidth = max(0.5,0.20 * frequency)
    X_filtered = gaussian_frequency_filter(X_centered,fs,frequency,bandwidth)
    pca = PCA()
    X_pca = pca.fit_transform(X_filtered)
    eigenvalue_spectrum[i, :] = pca.explained_variance_
    explained_variance_ratio[i, :] = pca.explained_variance_ratio_

out_path = 'out/freq'
if not os.path.exists(out_path):
    os.makedirs(out_path)

with open(out_path + '/expl_var_r_ieeg.p', "wb") as f:
    pickle.dump(explained_variance_ratio, f)
with open(out_path + '/expl_var_ieeg.p', "wb") as f:
    pickle.dump(eigenvalue_spectrum, f)


X_centered = data_matched
eigenvalue_spectrum = np.zeros((len(frequencies), X_centered.shape[0]))
explained_variance_ratio = np.zeros((len(frequencies), X_centered.shape[0]))

for i, frequency in enumerate(frequencies):
    print(frequency)
    bandwidth = max(0.5,0.20 * frequency)
    X_filtered = gaussian_frequency_filter(X_centered,fs,frequency,bandwidth)
    pca = PCA()
    X_pca = pca.fit_transform(X_filtered)
    eigenvalue_spectrum[i, :] = pca.explained_variance_
    explained_variance_ratio[i, :] = pca.explained_variance_ratio_

with open(out_path + '/expl_var_r_match.p', "wb") as f:
    pickle.dump(explained_variance_ratio, f)
with open(out_path + '/expl_var_match.p', "wb") as f:
    pickle.dump(eigenvalue_spectrum, f)