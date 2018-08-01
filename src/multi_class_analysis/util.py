import pickle
import numpy as np


np.random.seed(19680801)


def save_object(filename, regr):
    pickle.dump(regr, open(filename, "wb"))


def load_object(filename):
    regr = pickle.load(open(filename, "rb"))
    return regr
