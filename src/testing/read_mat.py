import scipy.io
mat = scipy.io.loadmat('105_label0.mat')
mat['time_stamps'].flatten()
