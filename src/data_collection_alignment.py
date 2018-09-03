from util import *
import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
from data_collection_merge_data import preprocess_dataframes, combine_data
from sklearn import preprocessing
import pickle


# filenames = "conv_head_1 conv_head_2 up_down_fast_1 up_down_fast_2 up_down_slow_1 up_down_slow_2 roll_slow_2 roll_slow_1 roll_fast_2 roll_fast_1 left_right_slow_2 left_right_slow_1 left_right_fast_2 left_right_fast_1".split()
# filename_to_index_list = {
#     'conv_head_1':[57, 73, 88, 102, 331, 604, 1469, 1604, 1806, 1831, 1885, 2143, 2432, 2737, 2765, 3000, 3570, 3791, 4307, 4339, 4361, 4707, 5666, 5731, 5896, 6045, 6090, 6133, 6347, 6442, 6628, 6821, 7233, 7272, 7741, 7869, 7891, 7921, 7960, 8098, 8130, 8194, 8256, 8366, 8522, 8538, 8778, 8991, 9075, 9111, 9199, 9229, 9311, 9404, 9477, 9526, 9584, 9696, 9750, 9936, 9949, 9960, 9974, 9985],
#     'conv_head_2':[28, 53, 65, 79, 90, 100, 177, 325, 386, 521, 626, 758, 910, 948, 1030, 1133, 1181, 1296, 1328, 1430, 1496, 1525, 1570, 1677, 1747, 1880, 1990, 2043, 2140, 2220, 2382, 2433, 2537, 2602, 2666, 2791, 2889, 2924, 2970, 3092, 3270, 3322, 3391, 3454, 3533, 3714, 3730, 3824, 3880, 4060, 4144, 4161, 4174, 4344, 4476, 4499, 4576, 4658, 4729, 4783, 4856, 4999, 5107, 5252, 5335, 5416, 5475, 5614, 5936, 6087, 6297, 6471, 6533, 6688, 6765, 6879, 6997, 7118, 7152, 7194, 7297, 7441, 7480, 7556, 7640, 7700, 7872, 7883, 7945, 7960, 8032, 8156, 8223, 8276, 8326, 8441, 8555, 8672, 8726, 8759, 8840, 8862, 8896, 9008, 9021, 9031, 9043, 9053],
#     'up_down_fast_1':[33, 61, 72, 86, 101, 112, 439, 475, 505, 615, 650, 676, 831, 864, 1077, 1128, 1251, 1324, 1375, 1552, 1863, 1891, 2055, 2086, 2110, 2412, 2436, 2457, 2660, 2703, 2747, 2932, 2970, 3075, 3115, 3238, 3468, 3487, 3590, 3612, 3721, 3748, 3867, 3892, 4059, 4088, 4865, 4883, 5330, 5503, 5544, 5678, 5693, 5705, 5720, 5732],
#     'up_down_fast_2':[25, 46, 56, 68, 79, 90, 588, 607, 626, 773, 810, 837, 1037, 1393, 1419, 1543, 1746, 1769, 1878, 1910, 2001, 2034, 2141, 2177, 2287, 2320, 2440, 2471, 2589, 2619, 2723, 2764, 3013, 3040, 3057, 3284, 3300, 3463, 3488, 3623, 3649, 3749, 3787, 3873, 4040, 4313, 4545, 4563, 4603, 4755, 4768, 4799, 5161, 5183, 5294, 5357, 5369, 5381, 5395, 5406],
#     'up_down_slow_1':[32, 91, 104, 118, 131, 146, 755, 951, 1269, 1299, 1322, 1501, 1716, 1747, 1969, 1991, 2112, 2244, 2609, 2638, 2797, 2965, 3048, 3142, 3288, 3519, 3549, 4051, 4077, 4755, 5316, 5335, 5437, 5448, 5459, 5472, 5482],
#     'up_down_slow_2':[82, 93, 107, 119, 131, 418, 441, 911, 933, 953, 976, 1143, 1168, 1401, 1454, 1600, 1841, 1866, 1926, 2163, 2195, 2449, 2473, 2742, 2762, 2985, 3016, 3262, 3295, 3362, 3452, 3677, 3771, 3844, 3951, 4077, 4114, 4224, 4644, 4669, 4688, 4714, 4745, 4994, 5056, 5067, 5078, 5092, 5101],
#     'roll_slow_2':[40, 50, 63, 75, 89, 275, 546, 571, 687, 810, 885, 1027, 1245, 1270, 1380, 1498, 1758, 2319, 2490, 2630, 2730, 2887, 3000, 3122, 3250, 3381, 3534, 3672, 3830, 4162, 4242, 4267, 4285, 4302, 4641, 4662, 4752, 4790, 4830, 5002, 5045, 5122, 5197, 5361, 5432, 5575, 5748, 5767, 5880, 5910, 6044, 6175, 6208, 6321, 6379, 6396, 6410, 6424, 6438],
#     'roll_slow_1':[42, 69, 82, 95, 108, 122, 340, 455, 505, 571, 678, 832, 975, 1212, 1254, 1491, 1505, 1543, 1690, 1787, 1865, 2036, 2114, 2177, 2231, 2288, 2458, 2483, 2500, 2668, 2690, 2710, 2806, 2917, 3026, 3069, 3090, 3112, 3202, 3307, 3333, 3540, 3723, 3851, 3883, 3914, 4447, 4634, 4943, 4973, 5002, 5139, 5393, 5609, 5621, 5633, 5646, 5658],
#     'roll_fast_2':[81, 95, 114, 130, 146, 330, 458, 559, 658, 762, 857, 979, 1095, 1190, 1286, 1360, 1428, 1442, 1507, 1617, 1714, 1795, 1839, 1940, 2027, 2125, 2241, 2340, 2455, 2559, 2646, 2729, 2764, 2903, 3052, 3162, 3385, 3414, 3443, 3608, 3765, 3780, 3962, 3996, 4019, 4137, 4156, 4297, 4413, 4531, 4640, 4750, 4862, 4968, 5082, 5196, 5272, 5305, 5421, 5556, 5704, 5718, 5733, 5747, 5759],
#     'roll_fast_1':[51, 65, 82, 96, 111, 351, 502, 527, 711, 740, 760, 945, 972, 1107, 1150, 1279, 1433, 1582, 1730, 1879, 2023, 2163, 2306, 2449, 2585, 2749, 2855, 3136, 3271, 3406, 3488, 3530, 3694, 3843, 4028, 4085, 4108, 4256, 4287, 4379, 4464, 4540, 4616, 4693, 4801, 4835, 4866, 4989, 5019, 5138, 5168, 5286, 5313, 5434, 5521, 5532, 5544, 5555, 5566],
#     'left_right_slow_2':[20, 30, 39, 50, 63, 281, 469, 651, 696, 883, 1073, 1280, 1314, 1513, 1689, 1734, 1905, 1964, 2126, 2173, 2338, 2350, 2366, 2497, 2637, 2664, 2801, 2904, 3027, 3190, 3215, 3406, 3432, 3453, 3606, 3628, 3767, 3789, 3911, 4028, 4064, 4190, 4229, 4360, 4392, 4508, 4548, 4678, 4717, 4813, 4873, 4985, 5008, 5106, 5141, 5247, 5383, 5441, 5488, 5498, 5511, 5524, 5535],
#     'left_right_slow_1':[45, 53, 66, 79, 89, 335, 373, 569, 795, 1066, 1266, 1309, 1444, 1544, 1889, 2163, 2328, 2410, 2612, 2858, 2892, 2963, 3087, 3213, 3225, 3245, 3455, 3504, 3679, 3718, 3747, 4005, 4062, 4274, 4325, 4467, 4527, 4827, 4842, 4866, 5086, 5114, 5228, 5406, 5461, 5596, 5611, 5626, 5641, 5654],
#     'left_right_fast_2':[22, 58, 69, 81, 98, 111, 318, 343, 361, 544, 579, 747, 788, 896, 1081, 1120, 1262, 1456, 1485, 1628, 1654, 1775, 1802, 1945, 2063, 2199, 2332, 2473, 2571, 2897, 2927, 2952, 2976, 3273, 3297, 3317, 3337, 3365, 3385, 3747, 3794, 3824, 3954, 4092, 4140, 4382, 4423, 4554, 4599, 4750, 4797, 4905, 4939, 5089, 5136, 5339, 5496, 5580, 5592, 5604, 5615, 5627],
#     'left_right_fast_1':[31, 72, 85, 96, 112, 122, 323, 343, 472, 597, 680, 879, 1108, 1250, 1379, 1484, 1611, 1705, 1795, 1821, 1923, 1960, 2033, 2153, 2252, 2325, 2376, 2467, 2592, 2696, 2772, 2884, 2989, 3157, 3350, 3394, 3456, 3506, 3554, 3683, 3813, 3923, 4085, 4190, 4307, 4341, 4480, 4572, 4634, 4700, 4810, 4895, 4985, 5020, 5039, 5117, 5131, 5174, 5252, 5286, 5366, 5402, 5488, 5566, 5579, 5592, 5607, 5619],
# }

subject_ids = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
label0_thresholds = {
    '101_label0':0.3,
    '102_label0':0.25,
    '103_label0':0.5,
    '104_label0':-1,  # file doesn't exist
    '105_label0':0.2,
    '106_label0':0.25,
    '107_label0':0.25,
    '108_label0':0.15,
    '109_label0':0.28,
    '110_label0':0.3,
    '111_label0':0.15,
    '112_label0':0.28,
    '113_label0':0.39,
    '114_label0':0.1,
    '115_label0':0.2,
    '116_label0':0.55,
    '117_label0':0.15
}
label1_thresholds = {
    '101_label1':-1,
    '102_label1':-1,
    '103_label1':-1,
    '104_label1':-1,
    '105_label1':-1,
    '106_label1':-1,
    '107_label1':-1,
    '108_label1':-1,
    '109_label1':-1,
    '110_label1':-1,
    '111_label1':-1,
    '112_label1':-1,
    '113_label1':-1,
    '114_label1':-1,
    '115_label1':-1,
    '116_label1':-1,
    '117_label1':-1
}
label2_thresholds = {
    '101_label2':-1,
    '102_label2':1.5,
    '103_label2':0.9,
    '104_label2':0.5,
    '105_label2':0.85,
    '106_label2':-1,
    '107_label2':0.85,
    '108_label2':0.92,
    '109_label2':0.75,
    '110_label2':0.73,
    '111_label2':1.13,
    '112_label2':0.87,
    '113_label2':1.01,
    '114_label2':0.82,
    '115_label2':0.64,
    '116_label2':0.93,
    '117_label2':0.88
}
label3_thresholds = {
    '101_label3':-1,
    '102_label3':1.25,
    '103_label3':2.18,
    '104_label3':0.49,
    '105_label3':0.97,
    '106_label3':1.0,
    '107_label3':-1,
    '108_label3':1.69,
    '109_label3':1.21,
    '110_label3':1.53,
    '111_label3':1.10,
    '112_label3':1.5,
    '113_label3':0.76,
    '114_label3':1.4,
    '115_label3':0.93,
    '116_label3':2.12,
    '117_label3':1.11
}
label4_thresholds = {
    '101_label4':-1,
    '102_label4':1.32,
    '103_label4':1.34,
    '104_label4':0.7,
    '105_label4':1.0,
    '106_label4':-1,
    '107_label4':1.17,
    '108_label4':1.57,
    '109_label4':0.85,
    '110_label4':1.08,
    '111_label4':0.55,
    '112_label4':0.71,
    '113_label4':1.0,
    '114_label4':1.0,
    '115_label4':0.87,
    '116_label4':1.19,
    '117_label4':0.94
}
label5_thresholds = {
    '101_label5':0.57,
    '102_label5':1.71,
    '103_label5':1.21,
    '104_label5':0.73,
    '105_label5':0.53,
    '106_label5':0.63,
    '107_label5':1.0,  # file exists, but pretend it doesn't, the data is trash
    '108_label5':1.3,
    '109_label5':0.78,
    '110_label5':1.22,
    '111_label5':0.78,
    '112_label5':1.2,
    '113_label5':1.1,
    '114_label5':0.77,
    '115_label5':1.43,
    '116_label5':1.0,
    '117_label5':0.56
}

if __name__ == '__main__':

    subject_id = subject_ids[4]
    path = "C:/Data_Experiment_W!NCE/{}/FACS/label0/".format(subject_id)
    jins_path = path+"jins/{}_label0.csv".format(subject_id)
    openface_path = path+"oface/{}_label0.csv".format(subject_id)
    output_path = path+"{}_alignments.pickle".format(subject_id)


    start_preview_size = 30  # how much time of the beginning should we save, by number of samples? (in seconds)
    end_preview_size = 60  # and the same for the end of the data

    eog_min = -400
    eog_max = 400
    of_min = 0
    of_max = 2.5

    print('loading jins data')
    # jins_df = pd.read_csv(jins_path, skiprows=5, nrows=100*preview_size_in_seconds)
    jins_df = pd.read_csv(jins_path, skiprows=5)

    print('loading openface data')
    # openface_df = pd.read_csv(openface_path, nrows=20*preview_size_in_seconds)
    openface_df = pd.read_csv(openface_path)


    print("normalizing openface input indices")
    num_jins_frames = jins_df.shape[0]
    num_openface_frames = openface_df.shape[0]

    def normalize_openface_frame(frame_index):
        return 1.0 * frame_index / num_openface_frames

    # filename_to_index_list[subject_id] = [normalize_openface_frame(x) for x in filename_to_index_list[subject_id]]


    print('preprocessing dataframes')
    jins_df, openface_df = preprocess_dataframes(jins_df, openface_df)

    jins_max_time = jins_df['TIME'].max()
    openface_max_time = openface_df['TIME'].max()

    def jins_time_to_normalized(jins_time):
        return jins_time/jins_max_time

    def jins_normalized_to_frame(jins_normalized):
        return int(round(jins_normalized*num_jins_frames))

    def openface_time_to_normalized(openface_time):
        return openface_time/openface_max_time

    def openface_normalized_to_frame(openface_normalized):
        return int(round(openface_normalized*num_openface_frames))

    # print('interpolating openface frames')
    # combined_df = combine_data(jins_df, openface_df)


    # print('[DEBUG] saving cleaned data')
    # jins_df.to_csv(jins_path+'.align.cleaned.csv')
    # openface_df.to_csv(openface_path+'.align.cleaned.csv')


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
            # check that this was a right click
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

        suptitle = 'Right click blue point then red point. Close window to save.'
        if 'first' in description:
            suptitle = '[START] '+suptitle
        else:
            suptitle = '[ENDING] '+suptitle

        fig.suptitle(suptitle)
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

    print("writing calibration frames to {}".format(output_path))
    with open(output_path, 'wb') as f:
        calibration_times = (jins_start_time, openface_start_time, jins_end_time, openface_end_time)
        pickle.dump(calibration_times, f)


    jins_normalized_start_time = jins_time_to_normalized(jins_start_time)
    jins_normalized_end_time = jins_time_to_normalized(jins_end_time)
    openface_normalized_start_time = openface_time_to_normalized(openface_start_time)
    openface_normalized_end_time = openface_time_to_normalized(openface_end_time)

    def normalize_normalize_openface(normalized_openface_time):
        return (normalized_openface_time-openface_normalized_start_time)/(openface_normalized_end_time-openface_normalized_start_time)

    def get_normalized_jins_time(normalized_normalized_openface_time):
        return jins_normalized_start_time + normalized_normalized_openface_time*(jins_normalized_end_time-jins_normalized_start_time)

    def openface_frame_to_jins_frame(openface_frame):
        openface_normalized = normalize_openface_frame(openface_frame)
        normalized_normalized_openface_time = normalize_normalize_openface(openface_normalized)
        jins_normalized = get_normalized_jins_time(normalized_normalized_openface_time)
        jins_frame = jins_normalized_to_frame(jins_normalized)
        return jins_frame


    # example_openface_frame = filename_to_index_list[subject_id][0]
    # example_openface_frame -= 1  # matlab is 1-indexed
    # example_jins_frame = openface_frame_to_jins_frame(example_openface_frame)
    # example_jins_frame += 1  # matlab is 1-indexed
    # print("openface -> jins: {} -> {}".format(example_openface_frame, example_jins_frame))
