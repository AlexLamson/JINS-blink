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


def get_jins_path(subject_number, label_number):
    return "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.csv".format(subject_number, label_number-1)


def get_jins_data(jins_path):
    jins_df = pd.read_csv(jins_path, skiprows=5)

    jins_df.drop(['//ARTIFACT'], axis=1, inplace=True)

    jins_df.rename( columns={'NUM': 'frame'}, inplace=True )  # rename the date column to time so it makes sense
    # jins_df['frame'] *= of_time_per_jins_time

    return jins_df


if __name__ == '__main__':

    subject_number = subject_numbers[0]
    label_number = label_numbers[0]

    for subject_number in subject_numbers:
        for label_number in label_numbers:
            jins_path = get_path(subject_number, label_number)
            if not file_exists(jins_path):
                print("MISSING "+jins_path)
                continue
            jins_df = get_jins_data(jins_path)
            # print(jins_df.shape)
            # exit()
            # pass

            exit()


    print("note: this file is still incomplete")
