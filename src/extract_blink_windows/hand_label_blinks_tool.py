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


def get_openface_path(subject_number, label_number):
    path1 = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/oface/{0}_label{1}.csv".format(subject_number, label_number)
    path2 = "C:/Data_Experiment_W!NCE/{0}/label{1}/oface/{0}_label{1}_1.csv".format(subject_number, label_number)
    if not file_exists(path1, sanitized=False) and file_exists(path2, sanitized=False):
        return path2
    return path1

def get_jins_path(subject_number, label_number):
    path1 = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.csv".format(subject_number, label_number)
    path2 = "C:/Data_Experiment_W!NCE/{0}/label{1}/jins/{0}_label{1}_1.csv".format(subject_number, label_number)
    if not file_exists(path1, sanitized=False) and file_exists(path2, sanitized=False):
        # print("path2: {}".format(path2))
        return path2
    # print("path1: {}".format(path2))
    return path1
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



            # if (subject_number == 101 and label_number == 5) or \
            if (subject_number == 100 and label_number == 5) or \
                    subject_number == 110 or \
                    (subject_number == 111 and label_number >= 4) or \
                    (subject_number == 112 and label_number == 5) or \
                    (subject_number == 113 and label_number == 2) or \
                    (subject_number == 114 and label_number == 3):
                print("({} {}) SKIPPING: hardcoded skip, data wasn't usable".format(subject_number, label_number))
                continue




            # skip subjects that already have had their blinks labeled
            blink_frames_filename = "blink_frames/blink_frames_{}_{}.txt".format(subject_number, label_number)
            if file_exists(blink_frames_filename):
                print("({} {}) SKIPPING: blinks already labeled".format(subject_number, label_number))
                continue



            # get the start times
            start_times_dict_string = '{}_label{}'.format(subject_number, label_number)
            has_start = True
            if start_times_dict_string not in start_times_dict:
                print("({} {}) NOTE: start time missing".format(subject_number, label_number))
                has_start = False
                # continue
            else:
                oface_start, jins_start = start_times_dict[start_times_dict_string]
                if oface_start < 0 or jins_start < 0:
                    print("({} {}) SKIPPING: start time is -1".format(subject_number, label_number))
                    continue
                oface_start, jins_start = oface_start-1, jins_start-1  # convert to zero-indexed
                # print("oface_start, jins_start: {} {}".format(oface_start, jins_start))

            # load in the openface data
            if has_start:
                openface_path = get_openface_path(subject_number, label_number)
                if not file_exists(openface_path, sanitized=False):
                    print("({} {}) SKIPPING: openface file missing".format(subject_number, label_number))
                    continue
                # print("openface_path: {}".format(openface_path))
                openface_df = pd.read_csv(openface_path)

            # load in the jins data
            jins_path = get_jins_path(subject_number, label_number)
            if not file_exists(jins_path, sanitized=False):
                print("({} {}) SKIPPING: jins file missing".format(subject_number, label_number))
                continue
            print("jins_path: {}".format(jins_path))
            jins_df = pd.read_csv(jins_path, skiprows=5)


            # clean up the dataframes
            if has_start:
                print("({} {}) LOADED".format(subject_number, label_number))
                jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)
            else:
                jins_df = preprocess_jins(jins_df)


            # drop initial frames to align data
            if has_start:
                openface_df = trim_by_start_frame(openface_df, oface_start)
                jins_df = trim_by_start_frame(jins_df, jins_start)

            # scale the jins data to try to eliminate the time inaccuracy problem
            jins_df['TIME'] *= of_time_per_jins_time


            # fill in the missing values in the openface data
            if has_start:
                interpolated_openface_data = np.interp(x=jins_df['TIME'], xp=openface_df['TIME'], fp=openface_df['AU45_r'])
                jins_df['AU45_r'] = pd.Series(interpolated_openface_data, index=jins_df.index)
            combined_df = jins_df


            # scale_factor = 10  # just for visualization purposes
            # combined_df['AU45_r'] = np.gradient(combined_df['AU45_r'])*scale_factor
            if start_times_dict_string in thresholds:
                threshold = thresholds[start_times_dict_string]
                # print("(threshold is {})".format(threshold))


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




            def get_points(combined_df, subject_number, label_number):
                # list of blink positions
                blink_frames = []

                # Simple mouse click function to store coordinates
                def onclick(event):
                    # check that this was a right click
                    if event.button == 3:
                        # global blink_frames
                        frame_number = int(np.round(event.xdata))
                        attention_grabber = ("# " if len(blink_frames) % 2 == 0 else " #")
                        print("Click detected at frame {} {} {}".format(frame_number, " "*30, attention_grabber*20))
                        blink_frames.append((frame_number))

                # print("plotting data")
                fig = plt.figure(1)
                fig.canvas.set_window_title("Right click on all the blinks")
                ax = fig.add_subplot(111)
                if has_start:
                    ax.set_xlim(left=jins_start, right=jins_start+1000)
                else:
                    ax.set_xlim(left=0, right=1000)
                ax.set_ylim(bottom=-2000, top=2000)
                fig.suptitle("Subject {} label {} ({})".format(subject_number, label_number, label_names[label_numbers.index(label_number)]))



                if has_start:
                    combined_df['AU45_r'] = combined_df['AU45_r'] * 500

                ax.plot(combined_df['frame'], combined_df['EOG_L'], alpha=0.5, label="EOG L")
                ax.plot(combined_df['frame'], combined_df['EOG_R'], alpha=0.5, label="EOG R")
                ax.plot(combined_df['frame'], combined_df['EOG_H'], alpha=0.5, label="EOG H")
                ax.plot(combined_df['frame'], combined_df['EOG_V'], alpha=0.5, label="EOG V")
                if has_start:
                    ax.plot(combined_df['frame'], combined_df['AU45_r'], label="AU45_r")

                # 'best'          0
                # 'upper right'   1
                # 'upper left'    2
                # 'lower left'    3
                # 'lower right'   4
                # 'right'         5
                # 'center left'   6
                # 'center right'  7
                # 'lower center'  8
                # 'upper center'  9
                # 'center'       10
                plt.legend(loc=3)



                # ax.plot(jins_df[['TIME']].values, eog_v_normalized, color='b', alpha=0.5)

                # ax.plot(jins_df[['TIME']].values, eog_v_normalized, color='b', alpha=0.5)
                # ax.plot(openface_df[['TIME']].values, au45_r_normalized, color='r', alpha=0.5)
                # ax.fill(openface_df[['TIME']].values, au45_r_normalized, facecolor='red', color='r', alpha=0.5)
                cid = fig.canvas.mpl_connect('button_press_event', onclick)  # add callback function

                # select the pan tool by default
                plt.Figure()
                thismanager = plt.get_current_fig_manager()
                thismanager.toolbar.pan()


                plt.show()



                fig.canvas.mpl_disconnect(cid)

                if len(blink_frames) == 0:
                    print("[EXIT] 0 points marked, exiting")
                    exit()

                return blink_frames


            blink_frames = get_points(combined_df, subject_number, label_number)
            blink_frames = sorted(blink_frames)
            blink_frames = np.array(blink_frames).astype(int)
            print(blink_frames)
            print("{} blinks collected".format(len(blink_frames)))
            # save_obj(blink_frames, blink_frames_filename)
            print("saving file to {}".format(blink_frames_filename))
            np.savetxt(fix_path(blink_frames_filename), blink_frames, fmt="%d", newline="\r\n")
            # exit()
