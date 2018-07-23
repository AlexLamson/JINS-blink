#!/usr/bin/python3
import sys
sys.path.append('..')
from util import *
from data_collection_merge_data import *
from openface_time_fixer import create_corrected_file

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd


def get_data(jins_path, openface_path, output_path):

    eog_min = -400
    eog_max = 400
    of_min = 0
    of_max = 2.5

    print('loading jins data')
    # jins_df = pd.read_csv(jins_filename, skiprows=5, nrows=100*preview_size_in_seconds)
    jins_df = pd.read_csv(jins_path, skiprows=5)

    print('loading openface data')
    # openface_df = pd.read_csv(openface_filename, nrows=20*preview_size_in_seconds)
    openface_df = pd.read_csv(openface_path)

    # print(openface_df)

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    # print(jins_df)
    # exit()

    # print('interpolating openface frames')
    # combined_df = combine_data(jins_df, openface_df)

    # print('[DEBUG] saving cleaned data')
    # jins_df.to_csv(jins_filename+'.align.cleaned.csv')
    # openface_df.to_csv(openface_filename+'.align.cleaned.csv')

    print('drop low quality eog frames')
    jins_df = jins_df.drop(jins_df[jins_df['EOG_V'] < eog_min].index)
    jins_df = jins_df.drop(jins_df[jins_df['EOG_V'] > eog_max].index)

    print('normalize & shift eog values')
    eog_v = jins_df[['EOG_V']].values.astype(float)
    # eog_v_normalized = eog_v
    eog_v_normalized = 2 * ((eog_v - eog_min)/(eog_max - eog_min) - 0.5)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # eog_v_normalized = min_max_scaler.fit_transform(eog_v)
    # eog_v_normalized = eog_v_normalized - 0.5

    print('normalize au45 values')
    au45_r = openface_df[['AU45_r']].values.astype(float)
    of_max = min(of_max, au45_r.max())
    au45_r_normalized = (au45_r - of_min)/(of_max - of_min)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # au45_r_normalized = min_max_scaler.fit_transform(au45_r)
    # au45_r_normalized = au45_r


    # eog_v_normalized = np.gradient(eog_v_normalized, axis=0)
    # eog_v_normalized = (eog_v_normalized) / (eog_v_normalized.max() - eog_v_normalized.min())


    jins_time = jins_df[['TIME']].values.astype(float)
    of_time = openface_df[['TIME']].values.astype(float)

    return jins_time, of_time, eog_v_normalized, au45_r_normalized


def start_aligner_tool(jins_path, openface_path, mat_path, output_path):
    jins_time, of_time, eog_v_normalized, au45_r_normalized = get_data(jins_path, openface_path, mat_path)

    jins_time *= of_time_per_jins_time

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    fig.canvas.set_window_title(output_path)  # set the window title

    # gracefully exit when someone presses the X button
    def handle_close(evt):
        print('user closed out of window, stopping program :)')
        exit()
    fig.canvas.mpl_connect('close_event', handle_close)


    initial_fine_tune = 0
    intial_shift_delta = 0

    # jins_plot, = plt.plot(jins_time, eog_v_normalized, lw=2, color='xkcd:blue')
    # of_plot, = plt.plot(of_time, au45_r_normalized, lw=2, color='xkcd:red')
    jins_plot, = plt.plot(jins_time, eog_v_normalized, lw=2, color='xkcd:cerulean')
    of_plot, = plt.plot(of_time, au45_r_normalized, lw=2, color='xkcd:light red')
    plt.axis([0, 45, -1, 1])

    axcolor = 'lightgoldenrodyellow'
    ax_shift_delta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    shift_delta_slider = Slider(ax_shift_delta, 'Shift data', -30, 10, valinit=intial_shift_delta)
    fine_tune_slider = Slider(axamp, 'Fine tuning', -1, 1, valinit=initial_fine_tune)


    def update(val):
        fine_tune = fine_tune_slider.val
        shift_delta = shift_delta_slider.val

        of_plot.set_xdata(of_time+shift_delta+fine_tune)

        fig.canvas.draw_idle()
    shift_delta_slider.on_changed(update)
    fine_tune_slider.on_changed(update)

    save_ax = plt.axes([0.7, 0.025, 0.2, 0.04])
    save_button = Button(save_ax, 'Save alignment', color=axcolor, hovercolor='0.975')


    def reset(event):
        print("you pressed the button")
        time_delta = shift_delta_slider.val + fine_tune_slider.val
        print("current time delta is: {}".format(time_delta))



        if time_delta < 0:
            jins_start_time = 0.0
            openface_start_time = -time_delta
        else:
            jins_start_time = time_delta
            openface_start_time = 0.0

        max_jins_time = (jins_time - jins_start_time).max()
        max_of_time = (of_time - openface_start_time).max()

        jins_end_time = min(max_jins_time, max_of_time)
        openface_end_time = min(max_jins_time, max_of_time)

        save_obj((jins_start_time, openface_start_time, jins_end_time, openface_end_time), output_path)


        # shift_delta_slider.reset()
        # fine_tune_slider.reset()
    save_button.on_clicked(reset)

    plt.show()


# # DEBUGGING
# # DEBUGGING
# path = '../../res/data5/'
# jins_filename, openface_filename = get_jins_openface_csv(path)
# output_path = "../../res/data5/alignment.pickle"
# # print(jins_filename)
# # exit()
# start_aligner_tool(path, jins_filename, openface_filename, output_path)
# output_path
# # DEBUGGING
# # DEBUGGING


for subject_id in all_folders_in_folder("C:/Data_Experiment_W!NCE/"):
    for label_id in all_folders_in_folder("C:/Data_Experiment_W!NCE/{0}/FACS".format(subject_id)):

        if not label_id.endswith('0'):
            print("skipping non-zero label")
            continue

        # generate paths to relevant files
        openface_path = "C:/Data_Experiment_W!NCE/{0}/FACS/{1}/oface/{0}_{1}.csv".format(subject_id, label_id)
        mat_path = "C:/Data_Experiment_W!NCE/{0}/FACS/{1}/time_stamps/{0}_{1}.mat".format(subject_id, label_id)
        corrected_openface_path = 'C:/Data_Experiment_W!NCE/{0}/FACS/{1}/oface/{0}_{1}_correct_times.csv'.format(subject_id, label_id)
        jins_path = "C:/Data_Experiment_W!NCE/{0}/FACS/{1}/jins/{0}_{1}.csv".format(subject_id, label_id)
        alignment_output_path = "C:/Data_Experiment_W!NCE/{0}/FACS/{1}/alignment.pickle".format(subject_id, label_id)

        # if we already aligned the file, don't bother trying to align it again
        if file_exists(alignment_output_path):
            print("skipping file that has already been aligned")
            continue

        # check that the files exist ( because sometimes they don't :( )
        if not file_exists(openface_path):
            print("subject {} {} missing openface data".format(subject_id, label_id))
            continue
        if not file_exists(mat_path):
            print("subject {} {} missing matlab timestamps".format(subject_id, label_id))
            continue
        if not file_exists(jins_path):
            print("subject {} {} missing jins data".format(subject_id, label_id))
            continue

        print("="*30)
        print("loading subject {} with {}".format(subject_id, label_id))
        print("="*30)
        '''
        correct the openface time stamps
        run the alignment script
        '''
        create_corrected_file(openface_path, mat_path, corrected_openface_path)
        start_aligner_tool(jins_path, corrected_openface_path, mat_path, alignment_output_path)
