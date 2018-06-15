import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from merge_data import preprocess_dataframes, combine_data
from sklearn import preprocessing
import pickle

'''
read in jins data
read in openface data

trim both to just the first 10 minutes of each

graph both

loop until you close out of the graph before completing the second click
    first click on graph, that's one peak on the EOG
    second click on graph, that's the matching peak on the regressor
    pad the start/ends of the dataframes such that the two points line up
    write to a file: the jins and openface times for the matching point
'''


if __name__ == '__main__':
    jins_fname = '../res/data1/jins_20180612174521.csv'
    openface_fname = '../res/data1/webcam_2018-06-12-13-45.csv'
    output_fname = '../res/data1/combined.csv'
    # jins_fname = '../res/data4/28A18305A891_20180612181409.csv'
    # openface_fname = '../res/data4/webcam_2018-06-12-14-12.csv'
    # output_fname = '../res/data4/combined.csv'

    preview_size_in_seconds = 1*60  # how much time of the beginning should we save, by number of samples?


    print('loading jins data')
    jins_df = pd.read_csv(jins_fname, skiprows=5, nrows=100*preview_size_in_seconds)

    print('loading openface data')
    openface_df = pd.read_csv(openface_fname, nrows=20*preview_size_in_seconds)

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    print('interpolating openface frames')
    combine_data(jins_df, openface_df)
    combined_df = jins_df

    # print('flatten the eog values to make it easier to read')
    # combined_df['EOG_V'] = combined_df['EOG_V'] - 0.5

    print('normalize & shift eog values')
    eog_v = combined_df[['EOG_V']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    eog_v_normalized = min_max_scaler.fit_transform(eog_v)
    eog_v_normalized = eog_v_normalized - 0.5

    print('normalize au45 values')
    au45_r = combined_df[['AU45_r']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    au45_r_normalized = min_max_scaler.fit_transform(au45_r)

    print("="*30)
    print("Your last 2 clicks should be on the blue point then the red point. Then you can close the window.")
    print("="*30)

    # Simple mouse click function to store coordinates
    def onclick(event):
        # figure out if this is a right click (later)

        global ix, iy
        ix, iy = event.xdata, event.ydata

        # print("{:.3f}, {:.3f}".format(ix, iy))

        # assign global variable to access outside of function
        global coords
        coords.append((ix, iy))

    coords = []

    print("plotting normalized data")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x = combined_df[['TIME']].values.astype(float)
    ax.plot(x, eog_v_normalized, color='b', alpha=0.5)
    ax.plot(x, au45_r_normalized, color='r', alpha=0.5)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)  # add callback function
    plt.show()

    fig.canvas.mpl_disconnect(cid)
    print("this is very important")

    # get the coordinates where the user clicked
    jins_x = coords[-2][0]
    openface_x = coords[-1][0]
    print("coords: {:.3f} {:.3f}".format(jins_x, openface_x))

    # determine the index of the each selected point
    jins_start_time = combined_df.ix[(combined_df['TIME']-jins_x).abs().argsort()[:2]].iloc[0]['TIME']
    openface_start_time = combined_df.ix[(combined_df['TIME']-openface_x).abs().argsort()[:2]].iloc[0]['TIME']
    print("jins_start_time: {:.3f} openface_start_time: {:.3f}".format(jins_start_time, openface_start_time))

    start_times_filename = "start_times.pickle"
    print("writing starting frames to {}".format(start_times_filename))
    with open(start_times_filename, 'wb') as f:
        start_times = (jins_start_time, openface_start_time)
        pickle.dump(start_times, f)
