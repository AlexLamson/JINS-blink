# replace the timestamp column with a matlab vector

import numpy as np
import pandas as pd
import scipy.io


def main():
    mat_fname = '../Data_Experiment_W!NCE/101/label0/time_stamps/101_label0_1.mat'
    open_face_fname = '../Data_Experiment_W!NCE/101/label0/oface/101_label0_1.csv'
    output_fname = '../Data_Experiment_W!NCE/101/label0/oface/101_label0_1_correct_times.csv'

    mat = scipy.io.loadmat(mat_fname)['time_stamps'].flatten()
    data = pd.read_csv(open_face_fname)

    data.iloc[:, 1] = mat

    data.to_csv(output_fname)


if __name__ == "__main__":
	main()
