import h5py
import sys
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from utils.config import Config
import os


def get_search_freq(N_total_samples, search_freq_left, search_freq_right, fs):
    """ Select the frequency points in the scanning frequency band """

    # Calculate the starting and end sample points
    t_start = 0
    t_end = N_total_samples / fs
    start_sample = math.floor(t_start * fs)
    end_sample = math.ceil(t_end * fs)

    # Select the frequency points in the scanning frequency band
    x_fr = fs / end_sample * np.arange(0, math.floor(end_sample / 2))
    freq_sels = list(np.where(np.logical_and(x_fr>=search_freq_left, x_fr<=search_freq_right)))[0]
    freq_sels = freq_sels.reshape((freq_sels.shape[0], 1))

    # Number of scanning frequency points
    N_freqs = len(freq_sels)

    # Frequency points to be scanned
    frequencies = x_fr[freq_sels]

    return start_sample, end_sample, freq_sels, N_freqs, frequencies


def developCSM(mic_signal, search_freq_left, search_freq_right, fs):
    """ Implements the generation of the cross-spectrum matrix (CSM) """

    # Total sample points
    N_total_samples = mic_signal.shape[0]

    # Number of microphone sensors
    N_mic = mic_signal.shape[1]

    # Select the frequency points in the scanning frequency band
    start_sample, end_sample,freq_sels, N_freqs, _ = get_search_freq(N_total_samples, search_freq_left, search_freq_right, fs)

    # Initialize the cross-spectrum matrix (CSM)
    CSM = np.zeros((N_mic, N_mic, N_freqs), dtype=complex)

    # Perform Fourier transform
    mic_signal_fft = np.sqrt(2) * fft(mic_signal[start_sample : end_sample + 1, :], axis=0) / (end_sample -  start_sample)

    # Develop CSM
    for K in range(0, N_freqs):
        # Calculate the CSM corresponding to the frequency K
        CSM[:, :, K] = mic_signal_fft[freq_sels[K], :].T * mic_signal_fft[freq_sels[K], :].conj()

    return CSM


def pyContourf(output, label, save_dir, file_name):
    """ Plotting the True and Estimated Beamforming Map """

    # Create figure
    fig = plt.figure(figsize=(41, 20))

    # Estimated beamforming map
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(output, cmap = plt.cm.hot)
    ax1.set_title('Estimated Beamforming Map')

    # Ideal beamforming map
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(label, cmap = plt.cm.hot)
    ax2.set_title('Ideal Beamforming Map')

    # Save figure
    plt.savefig(save_dir + '/' + file_name + '.png')
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # Initialize the AverageMeter by resetting all values
        self.reset()

    def reset(self):
        # Reset the stored values to start fresh
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Update the meter with a new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optimizer, opt, epoch, save_file):
    """ Saving model """

    # Create a dictionary to store the model's state, optimizer's state, options, and current epoch
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    # Save the state dictionary to the specified file
    torch.save(state, save_file)

    # Delete the state dictionary to free up memory
    del state


def get_microphone_info(micro_array_path):
    """Load microphone array data"""
    mic_array = h5py.File(micro_array_path, 'r')['array']

    # Get the x, y, and z coordinates
    mic_x_axis = mic_array[0]
    mic_x_axis = mic_x_axis.reshape(mic_x_axis.shape[0], 1)
    mic_y_axis = mic_array[1]
    mic_y_axis = mic_y_axis.reshape(mic_y_axis.shape[0], 1)                   
    mic_z_axis = np.zeros((mic_array.shape[1], 1))

    # Concatenate x, y, and z coordinates to form the microphone positions
    mic_pos = np.concatenate((mic_x_axis, mic_y_axis, mic_z_axis), axis=1)

    # Calculate the coordinates of the array center
    mic_centre = mic_pos.mean(axis=0)

    return mic_pos, mic_centre


def data_preprocess(raw_sound_data, yml_path):
    # Load configuration settings
    con = Config(yml_path).getConfig()['base']

    # Sampling frequency
    fs = con['fs']

    # Scanning frequency range
    scan_low_freq = con['scan_low_freq']
    scan_high_freq = con['scan_high_freq']

    # Develop CSM
    CSM = developCSM(raw_sound_data, scan_low_freq, scan_high_freq, fs)

    return CSM


class Logger(object):
    def __init__(self, fileN = "Default.log"):
        # Initialize the Logger
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        # Write the message to both the terminal and the log file
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def wgn(signal, snr):
    """ Generating Gaussian noise """

    # Convert the Signal-to-Noise Ratio (SNR) from dB to a linear scal
    snr = 10 ** (snr / 10.0)

    # Calculate the noise power based on the signal power and SNR
    random_index = np.random.randint(signal.shape[1])
    signal_power = np.sum(signal**2, axis=1)[random_index] / signal.shape[0]
    noise_power = signal_power / snr

    # Generate standard Gaussian noise
    noise = np.random.randn(signal.shape[0], signal.shape[1])

    return noise * np.sqrt(noise_power)


def neighbor_2_zero(matrix, source_x, source_y, n=2):
    # Adjust the source coordinates to ensure the neighborhood does not exceed boundaries
    if source_x-n < 0:
        source_x += n

    if source_y-n < 0:
        source_y += n

    if source_x+n >= matrix.shape[0]:
        source_x -= n

    if source_y+n >= matrix.shape[1]:
        source_y -= n

    # Set the values of the specified neighborhood to zero
    for i in range(source_x-n, source_x+n+1):
        for j in range(source_y-n, source_y+n+1):
            matrix[i][j] = 0

    return matrix


def find_match_source(output_mat_row, gt_mat):
    # Initialization
    min_index = -1
    location_bias = float("inf")

    # Iteratively find the sources
    for i in range(gt_mat.shape[0]):
        # Calculate the Euclidean distance (location bias)
        if location_bias > np.sqrt(((output_mat_row[0] - gt_mat[i][0])**2 + (output_mat_row[1] - gt_mat[i][1])**2)):
            location_bias = np.sqrt(((output_mat_row[0] - gt_mat[i][0])**2 + (output_mat_row[1] - gt_mat[i][1])**2))
            power_bias = np.abs(output_mat_row[2] - gt_mat[i][2])
            min_index = i

    # Remove the matched ground truth row from the matrix
    gt_mat = np.delete(gt_mat, min_index, 0)

    return min_index, location_bias, power_bias, gt_mat

