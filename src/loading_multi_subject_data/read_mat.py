import scipy.io
# mat = scipy.io.loadmat('105_label0.mat')
# print(mat['time_stamps'].flatten())

mat = scipy.io.loadmat('categories.mat')
print(mat['categories'].flatten())
