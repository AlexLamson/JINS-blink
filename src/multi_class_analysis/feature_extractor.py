import numpy as np
from numpy.linalg import norm
from scipy.stats import iqr
from math import sqrt
import scipy.io


'''
reference for motion features
=============================
https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions

Input shape
===========
[window frames * signals]

Signals
=======
[0] accel x
[1] accel y
[2] accel z
[3] gyro x
[4] gyro y
[5] gyro z
[6] eog l
[7] eog r
[8] eog h
[9] eog v
'''


global_signal_names = "accel x,accel y,accel z,gyro x,gyro y,gyro z,eog l,eog r,eog h,eog v".split(',')
# global_signal_names = "eog l,eog r,eog h,eog v".split(',')
# global_signal_names = "accel x,accel y".split(',')


def get_features(window, include_eog=True, include_imu=True):
    signal_names = global_signal_names.copy()
    features = []
    feature_names = []

    if not include_eog:
        window = window[:,:-4]
    if not include_imu:
        window = window[:,6:]

    # # compute motion magnitudes
    accel_magnitudes = norm(window[:,0:3], axis=1)
    window = np.concatenate((window, accel_magnitudes[:,np.newaxis]), axis=1)
    signal_names += ["accel mag"]

    gyro_magnitudes = norm(window[:,3:6], axis=1)
    window = np.concatenate((window, gyro_magnitudes[:,np.newaxis]), axis=1)
    signal_names += ["gyro mag"]


    # compute more signals
    original_signal_names = signal_names.copy()
    for i in range(len(original_signal_names)):
        # window = add_signal_derivative(window[:,i], original_signal_names[i], window, signal_names)
        # window = add_signal_cumsum(window[:,i], original_signal_names[i], window, signal_names)
        # window = add_signal_fft(window[:,i], original_signal_names[i], window, signal_names)
        pass


    # for i in [0, 1, 2, 3, 4, 5, 10, 11]:
    for i in [9, 10, 11]:
        window = add_signal_fft(window[:,i], original_signal_names[i], window, signal_names)
    # window = add_signal_fft(window[:,1], original_signal_names[i], window, signal_names)


    # compute some features from those signals
    for i in range(len(signal_names)):
        # print(window.shape)

        add_mean_crossings(window[:,i], signal_names[i], features, feature_names)
        add_std_dev(window[:,i], signal_names[i], features, feature_names)
        add_mean(window[:,i], signal_names[i], features, feature_names)
        add_max(window[:,i], signal_names[i], features, feature_names)
        add_min(window[:,i], signal_names[i], features, feature_names)
        # add_10_percentile(window[:,i], signal_names[i], features, feature_names)
        add_median(window[:,i], signal_names[i], features, feature_names)
        # add_90_percentile(window[:,i], signal_names[i], features, feature_names)
        # add_histogram(window[:,i], signal_names[i], features, feature_names)
        # add_median_absolute_deviation(window[:,i], signal_names[i], features, feature_names)
        add_energy(window[:,i], signal_names[i], features, feature_names)
        # add_variance(window[:,i], signal_names[i], features, feature_names)
        # add_iqr(window[:,i], signal_names[i], features, feature_names)

        # add_mean_crossings(window[:,i], signal_names[i], features, feature_names)
        # add_std_dev(window[:,i], signal_names[i], features, feature_names)
        # add_mean(window[:,i], signal_names[i], features, feature_names)
        # add_max(window[:,i], signal_names[i], features, feature_names)
        # add_min(window[:,i], signal_names[i], features, feature_names)
        # # add_10_percentile(window[:,i], signal_names[i], features, feature_names)
        # add_median(window[:,i], signal_names[i], features, feature_names)
        # # add_90_percentile(window[:,i], signal_names[i], features, feature_names)
        # # add_histogram(window[:,i], signal_names[i], features, feature_names)
        # # add_median_absolute_deviation(window[:,i], signal_names[i], features, feature_names)
        # add_energy(window[:,i], signal_names[i], features, feature_names)
        # # add_variance(window[:,i], signal_names[i], features, feature_names)
        # # add_iqr(window[:,i], signal_names[i], features, feature_names)


    # print("{} features: {}".format(len(feature_names), ", ".join([x for x in feature_names])))
    # print(len(feature_names))
    # exit()
    return features, feature_names


def add_signal_derivative(signal, signal_name, window, signal_names):
    derivative = np.gradient(signal)
    window = np.concatenate((window, derivative[:,np.newaxis]), axis=1)
    signal_names += ["derivative of {}".format(signal_name)]
    return window


def add_signal_cumsum(signal, signal_name, window, signal_names):
    cumsum = np.cumsum(signal)
    window = np.concatenate((window, cumsum[:,np.newaxis]), axis=1)
    signal_names += ["cumsum of {}".format(signal_name)]
    return window


def add_signal_fft(signal, signal_name, window, signal_names):
    fft = np.abs(np.fft.fft(signal))
    # print(fft)
    # print(fft.shape)
    
    # import matplotlib.pyplot as plt
    # plt.scatter(np.arange(len(fft)), fft)
    # plt.show()
    
    
    window = np.concatenate((window, fft[:,np.newaxis]), axis=1)
    signal_names += ["fft of {}".format(signal_name)]
    return window



def add_mean_crossings(signal, signal_name, features, feature_names):
    mean_adjusted_signal = signal - np.mean(signal)
    mean_crossings = ((mean_adjusted_signal[:-1] * mean_adjusted_signal[1:]) < 0).sum()
    # mean_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
    features.append(mean_crossings)
    feature_names.append( "# mean crossings of {}".format(signal_name) )


def add_std_dev(signal, signal_name, features, feature_names):
    std_dev = np.std(signal)
    features.append(std_dev)
    feature_names.append( "std dev of {}".format(signal_name) )


def add_mean(signal, signal_name, features, feature_names):
    mean = np.mean(signal)
    features.append(mean)
    feature_names.append( "mean of {}".format(signal_name) )


def add_max(signal, signal_name, features, feature_names):
    max_ = np.max(signal)
    features.append(max_)
    feature_names.append( "max of {}".format(signal_name) )


def add_min(signal, signal_name, features, feature_names):
    min_ = np.min(signal)
    features.append(min_)
    feature_names.append( "min of {}".format(signal_name) )


def add_10_percentile(signal, signal_name, features, feature_names):
    percentile = np.percentile(signal, 10)
    features.append(percentile)
    feature_names.append( "10th percentile of {}".format(signal_name) )


def add_median(signal, signal_name, features, feature_names):
    percentile = np.percentile(signal, 50)
    features.append(percentile)
    feature_names.append( "median of {}".format(signal_name) )


def add_90_percentile(signal, signal_name, features, feature_names):
    percentile = np.percentile(signal, 90)
    features.append(percentile)
    feature_names.append( "90th percentile of {}".format(signal_name) )


def add_median_absolute_deviation(signal, signal_name, features, feature_names):
    my_mad = mad(signal)
    features.append(my_mad)
    feature_names.append( "median absolute deviation of {}".format(signal_name) )


def add_energy(signal, signal_name, features, feature_names):
    energy = np.mean(signal**2)
    features.append(energy)
    feature_names.append( "energy of {}".format(signal_name) )


def add_variance(signal, signal_name, features, feature_names):
    variance = sqrt(np.mean(signal**2))
    features.append(variance)
    feature_names.append( "variance of {}".format(signal_name) )


def add_iqr(signal, signal_name, features, feature_names):
    interquartile_range = iqr(signal)
    features.append(interquartile_range)
    feature_names.append( "interquartile range of {}".format(signal_name) )


def add_histogram(signal, signal_name, features, feature_names):
    bins = 7
    hist = np.histogram(signal, bins=bins)[0]
    features.extend(hist)
    feature_names.extend( ["histogram of {} #{}/{}".format(signal_name, x+1, bins) for x in range(bins)] )


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
    # print("This file isn't meant to be run directly. Run train_machine_learning_model.py instead")

    subject_id, label_id = 101, 1
    path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.mat".format(subject_id, label_id)

    # [ trial * window frames * sensor channels ]
    subject_matrix = scipy.io.loadmat(path)['data_chunk']
    sample_window = subject_matrix[0, :, :]
    features, feature_names = get_features(sample_window)
    print("There are currently {} features being used".format(len(features)))
