import sys
sys.path.append('..')
from util import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from tqdm import tqdm
import scipy.io
from sklearn import preprocessing

from data_collection_merge_data import preprocess_dataframes, trim_by_start_time, trim_by_start_frame
from start_times import start_times_dict
from label_thresholds import thresholds
from aggregate_openface_blinks_data import *
from aggregate_jins_blinks_data import *


'''
Use the below variables to choose what data to give to the model
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
'''
# data_filename = "all_data_raw.pkl"
subject_numbers = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
label_numbers = [0, 2, 3, 4, 5]
label_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
of_time_per_jins_time = 1.00610973512302

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OPENFACE COLUMNS: frame TIME AU45_r
JINS COLUMNS: NUM TIME ACC_X ACC_Y ACC_Z GYRO_X GYRO_Y GYRO_Z EOG_L EOG_R EOG_H EOG_V
'''


if __name__ == '__main__':
    # print("running main")

    # iterate over all the subjects and labels
    for subject_number in subject_numbers:
        for label_number in label_numbers:


            # get the start times
            start_times_dict_string = '{}_label{}'.format(subject_number, label_number)
            has_start = True
            if start_times_dict_string not in start_times_dict:
                print("({} {}) SKIPPING: start time missing".format(subject_number, label_number))
                has_start = False
                oface_start, jins_start = 0, 0
                continue
            else:
                oface_start, jins_start = start_times_dict[start_times_dict_string]
                if oface_start < 0 or jins_start < 0:
                    print("({} {}) SKIPPING: start time is -1".format(subject_number, label_number))
                    continue
                oface_start, jins_start = oface_start-1, jins_start-1  # convert to zero-indexed
                # print("oface_start, jins_start: {} {}".format(oface_start, jins_start))




            # skip subjects that already have had their blinks labeled
            blink_frames_filename = "blink_frames/blink_frames_{}_{}.txt".format(subject_number, label_number)
            if file_exists(blink_frames_filename):
                print("({} {}) FIXING".format(subject_number, label_number))
                # subtract start time from each file
                blink_frames = np.loadtxt(fix_path(blink_frames_filename)).astype(int)

                blink_frames += jins_start

                with open(fix_path(blink_frames_filename), 'w') as f:
                    if len(blink_frames.shape) < 1:
                        f.write(str(blink_frames))
                        f.write('\n')
                    else:
                        for num in blink_frames:
                            f.write(str(num))
                            f.write('\n')
