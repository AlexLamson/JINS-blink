from util import *
import pandas as pd
# import numpy as np
# import util
import matplotlib.pyplot as plt
# from data_collection_merge_data import preprocess_dataframes, combine_data
# from sklearn import preprocessing
# import pickle


# matching_indices_filename = "matching_indices.pkl"
# matching_indices = load_obj(matching_indices_filename)
# # print(matching_indices)

# filenames = "conv_head_1 conv_head_2 up_down_fast_1 up_down_fast_2 up_down_slow_1 up_down_slow_2 roll_slow_2 roll_slow_1 roll_fast_2 roll_fast_1 left_right_slow_2 left_right_slow_1 left_right_fast_2 left_right_fast_1".split()

# for filename in filenames:
#     s = ", ".join([str(x) for x in matching_indices[filename]])
#     s = "["+s+"]"

#     print(filename+": "+s)


def plot_data(openface_path, label_id, subject_id):
    openface_df = pd.read_csv(openface_path)
    openface_df.columns = openface_df.columns.str.strip()  # columns have leading space, get rid of it

    # x=openface_df[['AU45_r']]
    # # x=list(openface_df.columns.values)
    # print(x)
    # exit()


    description = "label{} subject {}".format(label_id, subject_id)

    print("plotting data")
    fig = plt.figure(1)
    fig.canvas.set_window_title(description)
    ax = fig.add_subplot(111)
    # ax.set_xlim(left=left, right=right)
    # ax.set_ylim(bottom=-0.1, top=1.0)

    # ax2 = ax.twinx()

    ys = np.arange(0, 5, 0.01)
    plt.yticks(ys, [str(y) for y in ys])
    # plt.grid()

    suptitle = description
    # suptitle = 'Right click blue point then red point. Close window to save.'
    # if 'first' in description:
    #     suptitle = '[START] '+suptitle
    # else:
    #     suptitle = '[ENDING] '+suptitle

    fig.suptitle(suptitle)
    ax.plot(openface_df[['frame']].values, openface_df[['AU45_r']].values, color='r', alpha=0.9)
    plt.show()


found_it = False
resume_label = 1
resume_subject = 101

label_ids = [1, 2, 3, 4, 5]
subject_ids = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
for label_id in label_ids:
    for subject_id in subject_ids:
        if label_id == resume_label and subject_id == resume_subject:
            found_it = True
        if not found_it:
            continue

        # openface_path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/oface/{0}_label{1}.csv".format(subject_id, label_id)
        openface_path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/oface/{0}_label{1}_treadmill.csv".format(subject_id, label_id)

        if file_exists(openface_path):
            plot_data(openface_path, label_id, subject_id)
        else:
            print("file doesn't exist: "+openface_path)
