import sys
sys.path.append('..')
from util import *

import numpy as np
import scipy
from tqdm import tqdm
import scipy.io

from feature_extractor import get_features
from data_collection_merge_data import *


'''
Use the below variables to choose what data to give to the model
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
'''
subjects = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
labels = [1, 2, 3, 4, 5]
class_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')
'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

openface_path = path+"oface/{}_label{}.csv".format(subject_number, label_number-1)
openface_df = pd.read_csv(openface_path)

