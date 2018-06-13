import pandas as pd
import numpy as np


# this function is built very specifically for turning the jins time column into deltas
def time_column_to_delta(df):
    # only one decimal of precision is recorded for some reason, so the resolution is extrapolated
    # find the resolution - see how long the last digit is repeated for
    i = 0
    num = df['TIME'][i][-1]
    # find the first change
    while num == df['TIME'][i][-1]:
        i += 1
    num = df['TIME'][i][-1]
    start = i
    # find the second change
    while num == df['TIME'][i][-1]:
        i += 1
    # the number of frames inbetween the changes gives you the resolution
    res = 0.1 / (i - start)

    # the data is evenly spaced, so just apply the res time all the way down
    for i in range(df.shape[0]):
        df.at[i, 'TIME'] = i * res


# combines jins and openface frames into the jins dataframe.
def combine_data(jins_df, of_df, delay):
    jins_frame = 0
    jins_size_full = jins_df.shape[0]

    # find jins frame based on delay
    while jins_df['TIME'][jins_frame] < delay:
        jins_df.drop(jins_frame, inplace=True)
        jins_frame += 1

    last_of_time = of_df['timestamp'][of_df.shape[0] - 1]
    while jins_size_full > jins_frame and jins_df['TIME'][jins_frame] <= last_of_time:
        for column in of_df.columns:
            if column == 'timestamp':
                continue

            # interpolate the value of the column according to jins time
            jins_df.at[jins_frame, column] = np.interp(jins_df['TIME'][jins_frame] - delay, of_df['timestamp'], of_df[column])

        jins_frame += 1


if __name__ == '__main__':
    jins_fname = '../res/data1/jins_20180612174521.csv'
    openface_fname = '../res/data1/webcam_2018-06-12-13-45.csv'
    output_fname = '../res/data1/combined.csv'

    print('loading jins data')
    jins_df = pd.read_csv(jins_fname, skiprows=5)
    print('loading openface data')
    openface_df = pd.read_csv(openface_fname)
    print('done loading')

    jins_df.rename( columns={'DATE': 'TIME'}, inplace=True )  # rename the date column to time so it makes sense
    openface_df.columns = openface_df.columns.str.strip()  # columns have leading space, get rid of it

    openface_delay = 2.0  # seconds between the time JINS began recording and openface began recording

    # cull openface data
    # keep time, AU45_r (blink regression)
    openface_cols_to_drop = ['frame', 'face_id', 'confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r',
                             'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
                             'AU26_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c',
                             'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    openface_df.drop(openface_cols_to_drop, axis=1, inplace=True)

    # cull jins data
    # keep time and eog
    jins_cols_to_drop = ['//ARTIFACT', 'NUM', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    jins_df.drop(jins_cols_to_drop, axis=1, inplace=True)

    time_column_to_delta(jins_df)

    print('combining')
    combine_data(jins_df, openface_df, openface_delay)

    print('saving combined data')
    jins_df.to_csv(output_fname)
