import sys
sys.path.append('..')
from util import *

import numpy as np
import scipy
from tqdm import tqdm
import scipy.io

from data_collection_merge_data import *
from start_times import start_times_dict


'''
Use the below variables to choose what data to give to the model
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
'''
subject_numbers = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
label_numbers = [1, 2, 3, 4, 5]
class_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''


def get_path(subject_number, label_number):
    return "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/oface/{0}_label{1}.csv".format(subject_number, label_number-1)


def get_data(openface_path):
    openface_df = pd.read_csv(openface_path)
    # print(openface_df)

    # openface_df = openface_df[['frame','timestamp','AU45_r']]
    openface_df.columns = openface_df.columns.str.strip()  # columns have leading space, get rid of it
    openface_df = openface_df.filter(['frame','timestamp','AU45_r'], axis=1)
    openface_df = openface_df.set_index('frame')
    print(openface_df)

    # of_time_per_jins_time


if __name__ == '__main__':

    subject_number = subject_numbers[0]
    label_number = label_numbers[0]

    for subject_number in subject_numbers:
        for label_number in label_numbers:
            openface_path = get_path(subject_number, label_number)
            if not file_exists(openface_path):
                print("MISSING "+openface_path)
                continue
            openface_df = get_data(openface_path)
            # print(openface_df.shape)
            # exit()
            # pass

            exit()


    print("note: this file is still incomplete")
