print("loading libraries")

import pickle
import numpy as np
from os.path import isfile


np.random.seed(19680801)


def save_object(filename, regr):
    pickle.dump(regr, open(filename, "wb"))


def load_object(filename):
    regr = pickle.load(open(filename, "rb"))
    return regr


def file_exists(fname):
    return isfile(fname)
