import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn import preprocessing


# combines jins and openface frames into the jins dataframe.
def combine_data(jins_df, of_df):
    jins_frame = jins_df.iloc[0].name
    jins_size_full = jins_df.shape[0]
    end_of_time = of_df['TIME'].iloc[-1]

    with tqdm(total=jins_size_full) as pbar:
        while jins_frame < jins_size_full and jins_df['TIME'].iloc[jins_frame] <= end_of_time:
            for column in of_df.columns:
                if column == 'TIME':
                    continue

                # interpolate the value of the column according to jins time
                jins_df.at[jins_frame, column] = np.interp(jins_df['TIME'].iloc[jins_frame], of_df['TIME'], of_df[column])

            pbar.update(1)
            jins_frame += 1

    return jins_df

# # combines jins and openface frames into the jins dataframe.
# def combine_data(jins_df, of_df, delay=0):
#     jins_frame = 0
#     jins_size_full = jins_df.shape[0]

#     # find jins frame based on delay
#     while jins_df['TIME'][jins_frame] < delay:
#         jins_df.drop(jins_frame, inplace=True)
#         jins_frame += 1

#     last_of_time = of_df['timestamp'][of_df.shape[0] - 1]
#     while jins_size_full > jins_frame and jins_df['TIME'][jins_frame] <= last_of_time:
#         for column in of_df.columns:
#             if column == 'timestamp':
#                 continue

#             # interpolate the value of the column according to jins time
#             jins_df.at[jins_frame, column] = np.interp(jins_df['TIME'][jins_frame] - delay, of_df['timestamp'], of_df[column])

#         jins_frame += 1

#     return jins_df

# turns the jins time column into relative times
def time_column_to_delta(jins_df):
    # find the resolution by taking the difference of the first two timestamps
    first_time = float( jins_df['TIME'].iloc[0][-3:] )
    second_time = float( jins_df['TIME'].iloc[1][-3:] )
    res = second_time - first_time

    # the data is evenly spaced, so just apply the res time all the way down
    for i in range(jins_df.shape[0]):
        jins_df.at[i, 'TIME'] = i * res


def preprocess_dataframes(jins_df, openface_df):

    jins_df.rename( columns={'DATE': 'TIME'}, inplace=True )  # rename the date column to time so it makes sense
    openface_df.columns = openface_df.columns.str.strip()  # columns have leading space, get rid of it
    openface_df.rename( columns={'timestamp': 'TIME'}, inplace=True )  # rename the timestamp column so it's consistent

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

    # print('normalize jins data')
    # x = jins_df.values
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # jins_df = pd.DataFrame(x_scaled, columns=jins_df.columns)

    # print('normalize openface data')
    # x = openface_df.values
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # openface_df = pd.DataFrame(x_scaled, columns=openface_df.columns)

    return jins_df, openface_df


def trim_by_start_time(df, start_time):
    df = df.drop(df[df['TIME'] < start_time].index)
    df['TIME'] = df['TIME'] - df['TIME'].values[0]
    fresh_index = np.array(list(range(df.shape[0])))
    df.set_index(fresh_index, inplace=True)
    return df


if __name__ == '__main__':
    jins_fname = '../res/data1/jins_20180612174521.csv'
    openface_fname = '../res/data1/webcam_2018-06-12-13-45.csv'
    output_fname = '../res/data1/combined.csv'
    # jins_fname = '../res/data4/28A18305A891_20180612181409.csv'
    # openface_fname = '../res/data4/webcam_2018-06-12-14-12.csv'
    # output_fname = '../res/data4/combined.csv'

    print('loading jins data')
    jins_df = pd.read_csv(jins_fname, skiprows=5)

    print('loading openface data')
    openface_df = pd.read_csv(openface_fname)

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    print('aligning data')
    start_times_filename = "start_times.pickle"
    (jins_start_time, openface_start_time) = pickle.load(open(start_times_filename, "rb"))
    print("starting frames: jins {:.3f}  openface {:.3f}".format(jins_start_time, openface_start_time))
    openface_df = trim_by_start_time(openface_df, openface_start_time)
    jins_df = trim_by_start_time(jins_df, jins_start_time)

    print('saving cleaned data')
    jins_df.to_csv('../res/data1/jins_clean.csv')
    openface_df.to_csv('../res/data1/openface_clean.csv')

    print('combining (may be slow for first 10-15 seconds)')
    combine_data(jins_df, openface_df)

    print('saving combined data')
    jins_df.to_csv(output_fname)
