#!/usr/bin/python3
import sys
sys.path.append('..')
from util import *
import pandas as pd

jins_fname = '105_label0.csv'
jins_df = pd.read_csv(jins_fname, skiprows=5)
print(jins_df)
