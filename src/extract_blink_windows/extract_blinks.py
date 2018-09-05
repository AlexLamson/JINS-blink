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


'''
Use the below variables to choose what data to give to the model
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
'''
data_filename = "all_data_raw.pkl"
subject_numbers = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
label_numbers = [1, 2, 3, 4, 5]
label_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
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
    (X_all_raw, raw_index) = load_obj("../../res/all_data_raw.pkl", sanitized=False)
    print(raw_index.shape)

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

    print("note: this file is still incomplete")
