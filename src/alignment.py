from util import *
import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
from merge_data import preprocess_dataframes, combine_data
from sklearn import preprocessing
import pickle


if __name__ == '__main__':
    path = data_folder
    # path = '../res/data4/'
    output_fname = path + 'combined.csv'

    jins_fname, openface_fname = util.get_jins_openface_csv(path)

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

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

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

    coords = []

    def get_points(left, right, description='Choose first 2 calibration points'):
        # clear the recorded coords
        global coords
        coords = []

        if len(coords) > 0:
            print("there was a horrible problem. coords isn't getting cleared somehow")

        # Simple mouse click function to store coordinates
        def onclick(event):
            # figure out if this is a right click (later)
            if event.button == 3:
                global coords
                print("Point recorded at {:.3f} seconds".format(event.xdata))
                coords.append((event.xdata, event.ydata))

        print("plotting normalized data")
        fig = plt.figure(1)
        fig.canvas.set_window_title(description)
        ax = fig.add_subplot(111)
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=-0.1, top=1.0)

        fig.suptitle('Right click blue point then red point. Close window to save.')
        ax.plot(jins_df[['TIME']].values, eog_v_normalized, color='b', alpha=0.5)
        ax.plot(openface_df[['TIME']].values, au45_r_normalized, color='r', alpha=0.5)
        # ax.fill(openface_df[['TIME']].values, au45_r_normalized, facecolor='red', color='r', alpha=0.5)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)  # add callback function
        plt.show()

        fig.canvas.mpl_disconnect(cid)

        if len(coords) < 2:
            print("[ERROR] Not enough points clicked")
            exit()

        # get the coordinates where the user clicked
        jins_x = coords[-2][0]
        openface_x = coords[-1][0]
        print("coords: {:.3f} {:.3f}".format(jins_x, openface_x))

        return jins_x, openface_x

    print("collecting start times")
    jins_start_time, openface_start_time = get_points(left=0, right=start_preview_size, description='Choose first 2 calibration points')
    # jins_x_start, openface_x_start = get_points(left=0, right=start_preview_size, description='Choose first 2 calibration points')
    # # determine the index of the each selected point
    # jins_start_time = combined_df.ix[(combined_df['TIME']-jins_x_start).abs().argsort()[:2]].iloc[0]['TIME']
    # openface_start_time = combined_df.ix[(combined_df['TIME']-openface_x_start).abs().argsort()[:2]].iloc[0]['TIME']
    print("jins_start_time: {:.3f} openface_start_time: {:.3f}".format(jins_start_time, openface_start_time))

    print("collecting end times")
    max_time = max(jins_df[['TIME']].values.max(), openface_df[['TIME']].values.max())
    jins_end_time, openface_end_time = get_points(left=max_time-end_preview_size, right=max_time, description='Choose last 2 calibration points')
    # jins_x_end, openface_x_end = get_points(left=max_time-end_preview_size, right=max_time, description='Choose last 2 calibration points')
    # jins_end_time = combined_df.ix[(combined_df['TIME']-jins_x_end).abs().argsort()[:2]].iloc[0]['TIME']
    # openface_end_time = combined_df.ix[(combined_df['TIME']-openface_x_end).abs().argsort()[:2]].iloc[0]['TIME']
    print("jins_end_time: {:.3f} openface_end_time: {:.3f}".format(jins_end_time, openface_end_time))

    calibration_filename = "calibration_times.pickle"
    print("writing calibration frames to {}".format(calibration_filename))
    with open(calibration_filename, 'wb') as f:
        calibration_times = (jins_start_time, openface_start_time, jins_end_time, openface_end_time)
        pickle.dump(calibration_times, f)
