from util import *

import os
import pandas as pd


def get_all_data(data_folder_name):
    # iterate over each data folder
    trial_folders = all_folders_in_folder(data_folder_name)
    trial_folders = [os.path.join(data_folder_name, x) for x in trial_folders]
    trials_files = [x+'/combined.csv' for x in trial_folders]
    df_list = [pd.read_csv(trial_csv) for trial_csv in trials_files]

    # merge all the data files together
    big_df = pd.concat(df_list)

    return big_df

    # # old method using single trial
    # data_folder = '../res/data4/'
    # training_data_fname = data_folder+'combined.csv'
    # data = pd.read_csv(training_data_fname)

    # return data
