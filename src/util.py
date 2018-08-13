# set the random seed for repeatability
import numpy as np
rand_seed = 123456
np.random.seed(rand_seed)

# enable tqdm-pandas integration
from tqdm import tqdm
tqdm.pandas()

import os
from os import listdir
from os.path import isfile, isdir, join
import re
import pickle




data_folder = '../res/'

of_time_per_jins_time = 1.00610973512302


def get_jins_openface_csv(path):
    result_jins = None
    result_openface = None

    jins_re = '^[0-9|A]*_[0-9|A]*\.csv$'
    of_re    = '^webcam.*\.csv$'

    for fname in os.listdir(path):
        if re.match(jins_re, fname):
            if result_jins:
                print('WARNING: multiple candidates for jins file')
            result_jins = path + fname

        if re.match(of_re, fname):
            if result_openface:
                print('WARNING: multiple candidates for openface file')
            result_openface = path + fname

    return result_jins, result_openface


# if a file is open in Excel, it kills the program that tries to write to that file
# this function will let you try again if that's the case
def guarantee_execution(function, fargs):
    attempting = True

    while attempting:
        try:
            function(*fargs)
            attempting = False
        except Exception as e:
            print(e)

            if input("Try again? Submit 'n' to abort: ") == 'n':
                attempting = False


def fix_path(filename):
    if "/" in filename or "\\" in filename:
        return filename
    if not filename.startswith(data_folder):
        filename = data_folder + filename
    return filename


# http://stackoverflow.com/a/27518377/2230446
def get_num_lines(filename):
    filename = fix_path(filename)
    f = open(filename, "rb")
    num_lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        num_lines += buf.count(b"\n")
        buf = read_f(buf_size)

    return num_lines


# save a pickle file
def save_obj(obj, filename, print_debug_info=True):
    filename = fix_path(filename)
    sanitized_name = filename.replace('.pickle', '')
    with open(sanitized_name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if print_debug_info:
            print('Saved {}'.format(sanitized_name + '.pickle'))


# load a pickle file
def load_obj(filename, print_debug_info=True):
    filename = fix_path(filename)
    sanitized_name = filename.replace('.pickle', '')
    with open(sanitized_name + '.pickle', 'rb') as f:
        obj = pickle.load(f)
        if print_debug_info:
            print('Loaded {}'.format(sanitized_name + '.pickle'))
        return obj


def file_exists(filename):
    filename = fix_path(filename)
    return isfile(filename)


def all_files_in_folder(some_directory):
    only_files = [f for f in listdir(some_directory) if isfile(join(some_directory, f))]
    return only_files


def all_folders_in_folder(some_directory):
    only_files = [f for f in listdir(some_directory) if isdir(join(some_directory, f))]
    return only_files
