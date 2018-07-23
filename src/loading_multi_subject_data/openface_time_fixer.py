#!/usr/bin/python3
import sys
sys.path.append('..')
from util import *

# replace the timestamp column with a matlab vector

# import numpy as np
import pandas as pd
import scipy.io


def create_corrected_file(openface_path, mat_path, output_path):

    mat = scipy.io.loadmat(mat_path)['time_stamps'].flatten()
    data = pd.read_csv(openface_path)

    data.iloc[:, 1] = mat

    data.to_csv(output_path)


if __name__ == "__main__":
    subject_id, label_id = 101, "label0"

    openface_path = 'C:/Data_Experiment_W!NCE/{0}/FACS/{1}/oface/{0}_{1}.csv'.format(subject_id, label_id)
    mat_path = 'C:/Data_Experiment_W!NCE/{0}/FACS/{1}/time_stamps/{0}_{1}.mat'.format(subject_id, label_id)
    output_path = 'C:/Data_Experiment_W!NCE/{0}/FACS/{1}/oface/{0}_{1}_correct_times.csv'.format(subject_id, label_id)

    # create_corrected_file(openface_path, mat_path, output_path)
    guarantee_execution(create_corrected_file, (openface_path, mat_path, output_path))
