from util import *
import pandas as pd
import numpy as np
import pickle
import util
from tqdm import tqdm
from sklearn import preprocessing
from scipy.interpolate import interp1d


# pandas is actually terrible
pd.options.mode.chained_assignment = None


# combines jins and openface frames into the jins dataframe.
# def combine_data(jins_df, of_df):
def combine_data(jins_df, of_df, j_start, o_start, j_end, o_end):

    def openface_time_to_jins_time(of_time):
        normalized_time = (of_time-o_start)/(o_end-o_start)
        jins_time = normalized_time*(j_end-j_start) + j_start
        return jins_time

    # fix incorrect column dtype
    jins_df['TIME'] = jins_df['TIME'].astype(float)

    # trim the data
    jins_df = jins_df[jins_df['TIME'].between(j_start, j_end, inclusive=True)]
    of_df = of_df[of_df['TIME'].between(o_start, o_end, inclusive=True)]

    # try to warp the openface data to match the jins data
    of_df['TIME'] = of_df['TIME'].apply(openface_time_to_jins_time)

    print("warping openface data")
    interpolated_openface_data = np.interp(x=jins_df['TIME'], xp=of_df['TIME'], fp=of_df['AU45_r'])

    print('adding interpolated openface data to dataframe')
    jins_df['AU45_r'] = pd.Series(interpolated_openface_data, index=jins_df.index)

    return jins_df


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
    # some columns, like face_id, don't show up in every data set, so make sure we're only dropping cols that exist
    openface_cols_to_drop = list( set(openface_cols_to_drop) & set(openface_df.columns) )
    openface_df.drop(openface_cols_to_drop, axis=1, inplace=True)

    # cull jins data
    # keep time and eog
    jins_cols_to_drop = ['//ARTIFACT', 'NUM', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    jins_df.drop(jins_cols_to_drop, axis=1, inplace=True)

    time_column_to_delta(jins_df)

    return jins_df, openface_df


def trim_by_start_time(df, start_time):
    df = df.drop(df[df['TIME'] < start_time].index)
    df['TIME'] = df['TIME'] - df['TIME'].values[0]
    fresh_index = np.array(list(range(df.shape[0])))
    df.set_index(fresh_index, inplace=True)
    return df


if __name__ == '__main__':

    # path = '../res/data4/'
    path = data_folder
    output_fname = path + 'combined.csv'

    jins_fname, openface_fname = util.get_jins_openface_csv(path)

    print('loading jins data')
    jins_df = pd.read_csv(jins_fname, skiprows=5)

    print('loading openface data')
    openface_df = pd.read_csv(openface_fname)

    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    print('aligning data')
    start_times_filename = "calibration_times.pickle"
    (jins_start_time, openface_start_time, jins_end_time, openface_end_time) = pickle.load(open(start_times_filename, "rb"))
    print("starting frames: jins {:.3f}  openface {:.3f}".format(jins_start_time, openface_start_time))

    # print('[DEBUG] saving cleaned data')
    # jins_df.to_csv(jins_fname+'.cleaned.csv')
    # openface_df.to_csv(openface_fname+'.cleaned.csv')

    print('combining...')
    combined_df = combine_data(jins_df, openface_df, jins_start_time, openface_start_time, jins_end_time, openface_end_time)

    print('saving combined data')
    util.guarantee_execution(combined_df.to_csv, (output_fname,))
