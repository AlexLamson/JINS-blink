import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt





subjects = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
labels = [1, 2, 3, 4, 5]
class_names = "none,eyebrows lower,eyebrows raiser,cheek raiser,nose wrinkler,upper lip raiser,mouth open".split(',')

X_all_raw = None
X_all = None  # np.zeros(shape=(0,201*10))
y_all = []
groups = []


# accumulate the data for the all the subjects
print("reading raw data into memory")
for subject in tqdm(subjects):
    # subject_data = np.zeros(shape=(0,201,10))
    for label in labels:
        path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.mat".format(subject, label)

        # [ trial * window frames * sensor channels ]
        matlab_object = scipy.io.loadmat(path)

        subject_matrix = matlab_object['data_chunk']
        # subject_data = np.concatenate((subject_data, subject_matrix), axis=0)
        groups += [subject]*subject_matrix.shape[0]
        y_all += [label]*subject_matrix.shape[0]

        for trial in range(subject_matrix.shape[0]):
            raw_window = subject_matrix[trial,:,:]
            # print(raw_window.shape)
            if X_all_raw is None:
                X_all_raw = np.empty(shape=(0,len(raw_window), 10), dtype=float)
            # print(X_all_raw.shape)
            # exit()
            X_all_raw = np.concatenate((X_all_raw, raw_window[np.newaxis,:,:]), axis=0)


print("normalizing data")
# normalize accelerometer signals
a = np.mean(np.std(X_all_raw[:,:,0:3], axis=2))
X_all_raw[:,:,0:3] = X_all_raw[:,:,0:3] / a

# normalize gyroscope signals
a = np.mean(np.std(X_all_raw[:,:,3:6], axis=2))
X_all_raw[:,:,3:6] = X_all_raw[:,:,3:6] / a

# normalize eog signals
a = np.mean(np.std(X_all_raw[:,:,6:], axis=2))
X_all_raw[:,:,6:10] = X_all_raw[:,:,6:10] / a





some_arbitrary_trial = 0   # -13#125#50
print("plotting window for trial at index {}".format(some_arbitrary_trial))



single_window = X_all_raw[some_arbitrary_trial,:,:]

x = np.arange(single_window.shape[0])

fig = plt.figure("Trial #{}".format(some_arbitrary_trial))
plt.title("Trial #{}".format(some_arbitrary_trial))

# accelerometer
plt.scatter(x, single_window[:,0], c='xkcd:red', alpha=0.5, label="accel x")
plt.scatter(x, single_window[:,1], c='xkcd:orange', alpha=0.5, label="accel y")
plt.scatter(x, single_window[:,2], c='xkcd:goldenrod', alpha=0.5, label="accel z")

# gyroscope
plt.scatter(x, single_window[:,3], c='xkcd:green', alpha=0.5, label="gyro x")
plt.scatter(x, single_window[:,4], c='xkcd:blue', alpha=0.5, label="gyro y")
plt.scatter(x, single_window[:,5], c='xkcd:indigo', alpha=0.5, label="gyro z")

# eog signals
plt.scatter(x, single_window[:,6], c='xkcd:violet', alpha=0.5, label="eog l")
plt.scatter(x, single_window[:,7], c='xkcd:gray', alpha=0.5, label="eog r")
plt.scatter(x, single_window[:,8], c='xkcd:black', alpha=0.5, label="eog h")
plt.scatter(x, single_window[:,9], c='xkcd:bright pink', alpha=0.5, label="eog v")

plt.xlabel("Frame #")
plt.ylabel("Magnitude of feature")
plt.legend(loc=2)
plt.show()
