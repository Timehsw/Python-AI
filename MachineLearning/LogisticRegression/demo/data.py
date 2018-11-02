import pandas as pd, numpy as np
import random

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

def to_label(data, target, percentile):
	'''
	Input: data, name of target column, percentile to partition data
	Output: data, but with the target column values
	changed from continuous to categorial (classes)
	'''
	frac = percentile / 100.0
	part_val = data[target].quantile(frac)
	data[target] = [1 if d > part_val else 0 for d in data[target]]
	return data

def load_data(filename):
    if filename == 'digits':
    	# Load in the data
    	digits = load_digits()
    	data = pd.DataFrame(digits.data)
    	label = 'class'
    	data[label] = digits.target

    elif filename == 'boston':
    	# Load in the data
    	boston = load_boston()
    	data = pd.DataFrame(boston.data, columns=boston.feature_names)
    	label = 'HomeVal50'
    	data[label] = boston.target
    	# Transform the target variable to binary
    	to_label(data, label, 50)

    else:
    	print("Please enter a valid file name.")

    return data, label
