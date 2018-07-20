#!/usr/bin/python3
import sys
sys.path.append('..')
from util import *
from data_collection_merge_data import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd



def get_data():
    # path = data_folder
    path = '../../res/data5/'
    output_fname = path + 'combined.csv'

    jins_fname, openface_fname = get_jins_openface_csv(path)

    start_preview_size = 30  # how much time of the beginning should we save, by number of samples? (in seconds)
    end_preview_size = 60  # and the same for the end of the data

    eog_min = -400
    eog_max = 400
    of_min = 0
    of_max = 2.5

    print('loading jins data')
    # jins_df = pd.read_csv(jins_fname, skiprows=5, nrows=100*preview_size_in_seconds)
    jins_df = pd.read_csv(jins_fname, skiprows=5)

    print('loading openface data')
    # openface_df = pd.read_csv(openface_fname, nrows=20*preview_size_in_seconds)
    openface_df = pd.read_csv(openface_fname)

    # print(openface_df)

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    # print(jins_df)
    # exit()

    # print('interpolating openface frames')
    # combined_df = combine_data(jins_df, openface_df)

    # print('[DEBUG] saving cleaned data')
    # jins_df.to_csv(jins_fname+'.align.cleaned.csv')
    # openface_df.to_csv(openface_fname+'.align.cleaned.csv')

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


    jins_time = jins_df[['TIME']].values.astype(float)
    of_time = openface_df[['TIME']].values.astype(float)

    return jins_time, of_time, eog_v_normalized, au45_r_normalized


jins_time, of_time, eog_v_normalized, au45_r_normalized = get_data()




fig, ax = plt.subplots()
plt.subplots_adjust(left=0.0, bottom=0.25)

initial_fine_tune = 0
intial_shift_delta = 0

jins_plot, = plt.plot(jins_time, eog_v_normalized, lw=2, color='blue')
of_plot, = plt.plot(of_time, au45_r_normalized, lw=2, color='red')
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

    # shift_delta_slider.reset()
    # fine_tune_slider.reset()
save_button.on_clicked(reset)

plt.show()
