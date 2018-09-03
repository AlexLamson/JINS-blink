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

'''
The thresholds are recording using the derivative of the OpenFace AU45_r signal.
Anywhere the derivative of AU45_r is higher than the given threshold, treat that as a blink.
'''
label0_thresholds = {
    '101_label0':0.3,
    '102_label0':0.25,
    '103_label0':0.5,
    '104_label0':-1,  # file doesn't exist
    '105_label0':0.2,
    '106_label0':0.25,
    '107_label0':0.25,
    '108_label0':0.15,
    '109_label0':0.28,
    '110_label0':0.3,
    '111_label0':0.15,
    '112_label0':0.28,
    '113_label0':0.39,
    '114_label0':0.1,
    '115_label0':0.2,
    '116_label0':0.55,
    '117_label0':0.15
}
label1_thresholds = {
    '101_label1':-1,
    '102_label1':-1,
    '103_label1':-1,
    '104_label1':-1,
    '105_label1':-1,
    '106_label1':-1,
    '107_label1':-1,
    '108_label1':-1,
    '109_label1':-1,
    '110_label1':-1,
    '111_label1':-1,
    '112_label1':-1,
    '113_label1':-1,
    '114_label1':-1,
    '115_label1':-1,
    '116_label1':-1,
    '117_label1':-1
}
label2_thresholds = {
    '101_label2':-1,
    '102_label2':1.5,
    '103_label2':0.9,
    '104_label2':0.5,
    '105_label2':0.85,
    '106_label2':-1,
    '107_label2':0.85,
    '108_label2':0.92,
    '109_label2':0.75,
    '110_label2':0.73,
    '111_label2':1.13,
    '112_label2':0.87,
    '113_label2':1.01,
    '114_label2':0.82,
    '115_label2':0.64,
    '116_label2':0.93,
    '117_label2':0.88
}
label3_thresholds = {
    '101_label3':-1,
    '102_label3':1.25,
    '103_label3':2.18,
    '104_label3':0.49,
    '105_label3':0.97,
    '106_label3':1.0,
    '107_label3':-1,
    '108_label3':1.69,
    '109_label3':1.21,
    '110_label3':1.53,
    '111_label3':1.10,
    '112_label3':1.5,
    '113_label3':0.76,
    '114_label3':1.4,
    '115_label3':0.93,
    '116_label3':2.12,
    '117_label3':1.11
}
label4_thresholds = {
    '101_label4':-1,
    '102_label4':1.32,
    '103_label4':1.34,
    '104_label4':0.7,
    '105_label4':1.0,
    '106_label4':-1,
    '107_label4':1.17,
    '108_label4':1.57,
    '109_label4':0.85,
    '110_label4':1.08,
    '111_label4':0.55,
    '112_label4':0.71,
    '113_label4':1.0,
    '114_label4':1.0,
    '115_label4':0.87,
    '116_label4':1.19,
    '117_label4':0.94
}
label5_thresholds = {
    '101_label5':0.57,
    '102_label5':1.71,
    '103_label5':1.21,
    '104_label5':0.73,
    '105_label5':0.53,
    '106_label5':0.63,
    '107_label5':1.0,  # file exists, but pretend it doesn't, the data is trash
    '108_label5':1.3,
    '109_label5':0.78,
    '110_label5':1.22,
    '111_label5':0.78,
    '112_label5':1.2,
    '113_label5':1.1,
    '114_label5':0.77,
    '115_label5':1.43,
    '116_label5':1.0,
    '117_label5':0.56
}

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

