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
label_numbers = [1, 2, 3, 4, 5]
label_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
of_time_per_jins_time = 1.00610973512302
'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OPENFACE COLUMNS: frame TIME AU45_r
JINS COLUMNS: NUM TIME ACC_X ACC_Y ACC_Z GYRO_X GYRO_Y GYRO_Z EOG_L EOG_R EOG_H EOG_V
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

    # iterate over all the subjects and labels
    for subject_number in subject_numbers:
        for label_number in label_numbers:

            # if label_number < 2 or subject_number < 103:
            #     continue


            # get the start times
            start_times_dict_string = '{}_label{}'.format(subject_number, label_number)
            if start_times_dict_string not in start_times_dict:
                print("({} {}) SKIPPING: start time missing".format(subject_number, label_number))
                continue
            oface_start, jins_start = start_times_dict[start_times_dict_string]
            if oface_start < 0 or jins_start < 0:
                print("({} {}) skipping: start time is -1".format(subject_number, label_number))
                continue
            oface_start, jins_start = oface_start-1, jins_start-1  # convert to zero-indexed
            print("oface_start, jins_start: {} {}".format(oface_start, jins_start))

            # load in the openface data
            openface_path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/oface/{0}_label{1}.csv".format(subject_number, label_number)
            if not file_exists(openface_path, sanitized=False):
                print("({} {}) SKIPPING: openface file missing".format(subject_number, label_number))
                continue
            print("openface_path: {}".format(openface_path))
            openface_df = pd.read_csv(openface_path)

            # load in the jins data
            jins_path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.csv".format(subject_number, label_number)
            if not file_exists(jins_path, sanitized=False):
                print("({} {}) SKIPPING: jins file missing".format(subject_number, label_number))
                continue
            print("jins_path: {}".format(jins_path))
            jins_df = pd.read_csv(jins_path, skiprows=5)


            # clean up the dataframes
            print("({} {}) LOADED".format(subject_number, label_number))
            jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)


            # drop initial frames to align data
            openface_df = trim_by_start_frame(openface_df, oface_start)
            jins_df = trim_by_start_frame(jins_df, jins_start)

            # scale the jins data to try to eliminate the time inaccuracy problem
            jins_df['TIME'] *= of_time_per_jins_time


            # fill in the missing values in the openface data
            interpolated_openface_data = np.interp(x=jins_df['TIME'], xp=openface_df['TIME'], fp=openface_df['AU45_r'])
            jins_df['AU45_r'] = pd.Series(interpolated_openface_data, index=jins_df.index)
            combined_df = jins_df


            scale_factor = 10  # just for visualization purposes
            # combined_df['AU45_r'] = np.gradient(combined_df['AU45_r'])*scale_factor
            threshold = thresholds[start_times_dict_string]
            print("threshold is {}".format(threshold))

            
            # mask = combined_df['AU45_r'] > threshold
            # combined_df['AU45_r'][mask] = 5
            # combined_df['AU45_r'][np.invert(mask)] = 0


            # combined_df['AU45_r'] = np.gradient(combined_df['AU45_r'])
            # blink_starts = np.argwhere(combined_df['AU45_r'] > 0).flatten()[0::2]
            # blink_ends = np.argwhere(combined_df['AU45_r'] < 0).flatten()[0::2]
            # blink_pairs = np.array(list(zip(blink_starts, blink_ends)))
            # blink_indicies = np.mean(blink_pairs, axis=1).astype(int)
            # # print(blink_indicies)
            # # print(blink_starts)
            # # print(blink_ends)

            # combined_df['peaks'] = np.zeros(len(jins_df['AU45_r']))
            # combined_df['peaks'][blink_indicies] = 5


            # exit()




            # import peakutils
            # indexes = peakutils.indexes(combined_df['AU45_r'], thres=threshold, min_dist=15)

            # new_column = np.zeros(len(jins_df['AU45_r']))
            # new_column[indexes] = 5
            # jins_df['peaks'] = pd.Series(new_column, index=jins_df.index)



            # print(a)
            # exit()


            # a=combined_df['AU45_r']
            # print(a.shape)
            # a=np.gradient(combined_df['AU45_r'])
            # print(a.shape)
            # exit()


            '''
            OPENFACE COLUMNS: frame TIME AU45_r
            JINS COLUMNS: frame TIME ACC_X ACC_Y ACC_Z GYRO_X GYRO_Y GYRO_Z EOG_L EOG_R EOG_H EOG_V
            '''
            # TODO: normalize the data

            # print(combined_df)
            # exit()


            # combined_df['EOG_V'] /= np.std(combined_df['EOG_V'])
            # plt.plot(combined_df['TIME'], combined_df['EOG_V'])
            # plt.plot(combined_df['TIME'], combined_df['peaks'])
            # plt.plot(combined_df['TIME'], combined_df['AU45_r'])

            # combined_df['peaks'] = combined_df['peaks'] * 500
            combined_df['AU45_r'] = combined_df['AU45_r'] * 500

            plt.plot(combined_df['frame'], combined_df['EOG_L'])
            plt.plot(combined_df['frame'], combined_df['EOG_R'])
            plt.plot(combined_df['frame'], combined_df['EOG_H'])
            plt.plot(combined_df['frame'], combined_df['EOG_V'])
            plt.plot(combined_df['frame'], combined_df['AU45_r'])
            # plt.plot(combined_df['frame'], combined_df['peaks'])

            plt.show()


            exit()


    #       find all the positive (negative to positive) crossings
    #       find all the negative (positive to negative) crossings
    #       pair them up and take the average of each pair to get the x location of each peak

    #       for each blink point
    #           select a window of data starting 6 frames before to 6 frames after
    #           save the window to list of windows (keep which subject & label it came from)

    # output the list of blinks as a csv file

    print("note: this file is still incomplete")
