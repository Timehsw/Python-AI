import pandas as pd, numpy as np
import random

def cross_val_split(data, folds, index):
	'''
	Function to split the data into train-test sets for k-fold cross-validation
	'''
	data_idx = []
	indices = [i for i in range(data.shape[0])]

	fold_size = int(len(data)/folds)
	for i in range(folds):
	    fold_idx = []
	    while len(fold_idx) < fold_size:
	        idx = random.randrange(len(indices))
	        fold_idx.append(indices.pop(idx))
	    data_idx.append(fold_idx)

	test_idx = data_idx[index]
	del data_idx[index]
	train_idx = [item for sublist in data_idx for item in sublist]

	test = data.iloc[test_idx]
	train = data.iloc[train_idx]

	return train,test

def train_test_split(data, label, test_ratio=0.2):
	'''
	Fuction to split the dataset into train-test sets
	based on the specified size of the test set
	'''
	test_idx = []
	indices = [i for i in range(data.shape[0])]

	test_size = test_ratio * len(data)
	while len(test_idx) < test_size:
		test_idx.append(random.randrange(len(indices)))

	train_idx = [i for i in indices if i not in test_idx]

	test = data.iloc[test_idx]
	train = data.iloc[train_idx]

	y_train = train[label]
	X_train = train.drop(label,axis=1)

	y_test = test[label]
	X_test = test.drop(label,axis=1)

	return X_train, y_train, X_test, y_test
