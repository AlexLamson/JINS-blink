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

from start_times import start_times_dict
from label_thresholds import *
from aggregate_openface_blinks_data import *
from aggregate_jins_blinks_data import *


'''
Use the below variables to choose what data to give to the model
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
'''
# data_filename = "all_data_raw.pkl"
subject_numbers = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
label_numbers = [1, 2, 3, 4, 5]
label_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
of_time_per_jins_time = 1.00610973512302
'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''


# load the jins data
# load the openface data
# create functions to translate the timestamps between the two

# for each subject
#     for each label
#         if it has a threshold (-1 doesn't count)
#             find all the positive (negative to positive) crossings
#             find all the negative (positive to negative) crossings
#             pair them up and take the average of each pair to get the x location of each peak

#             for each blink point
#                 select a window of data starting 6 frames before to 6 frames after
#                 save the window to list of windows (keep which subject & label it came from)

# output the list of blinks as a csv file


if __name__ == '__main__':
    print("running main")

    # load the jins data
    # (X_all_raw, raw_index) = load_obj("../../res/all_data_raw.pkl", sanitized=False)
    '''
    raw_index: [subject number, label number, trial number]
    X_all_raw: [ trial, window frames, sensor channels ]
    '''

    # print(raw_index.shape)

    # print(raw_index)
    # exit()

    # iterate over all the subjects and labels
    for subject_number in subject_numbers:
        for label_number in label_numbers:

            # get the start times
            start_times_dict_string = '{}_label{}'.format(subject_number, label_number)
            if start_times_dict_string not in start_times_dict:
                print("({} {}) skipping: start time missing".format(subject_number, label_number))
                continue
            oface_start, jins_start = start_times_dict[start_times_dict_string]
            if oface_start < 0 or jins_start < 0:
                print("({} {}) skipping: start time is -1".format(subject_number, label_number))
                continue
            oface_start, jins_start = oface_start-1, jins_start-1  # convert to zero-indexed

            # load the jins data
            jins_path = get_jins_path(subject_number, label_number)
            if not file_exists(jins_path, sanitized=False):
                print("({} {}) skipping: jins data is missing".format(subject_number, label_number))
                continue
            print("({} {}) loading".format(subject_number, label_number))
            jins_df = get_jins_data(jins_path)  # read the jins file
            # print(jins_df)
            # exit()

            # load the openface data
            openface_path = get_openface_path(subject_number, label_number)
            if not file_exists(openface_path, sanitized=False):
                print("({} {}) skipping: openface data is missing".format(subject_number, label_number))
                continue
            print("({} {}) loading".format(subject_number, label_number))
            openface_df = get_openface_data(openface_path)  # read the openface file
            # print(openface_df.shape)
            # exit()
            # pass

            a = openface_df['frame'].iloc[-10:]
            b = jins_df['frame'].iloc[-10:]
            print("openface: ")
            print(a)
            print("jins: ")
            print(b)
            exit()
            pass


            # a = openface_df['AU45_r']
            # print(a)
            print(openface_df.iloc[oface_start:])
            exit()



            # create functions to translate the timestamps between the two

            exit()


    # for each subject
    #     for each label
    #         if it has a threshold (-1 doesn't count)
    #             find all the positive (negative to positive) crossings
    #             find all the negative (positive to negative) crossings
    #             pair them up and take the average of each pair to get the x location of each peak

    #             for each blink point
    #                 select a window of data starting 6 frames before to 6 frames after
    #                 save the window to list of windows (keep which subject & label it came from)

    # output the list of blinks as a csv file

    print("note: this file is still incomplete")
