import sys

import pandas as pd, numpy as np
import random
import pickle

from MachineLearning.LogisticRegression.demo.train_test_split import train_test_split
from MachineLearning.LogisticRegression.demo.data import load_data


class LogisticRegression():
    def __init__(self, X, y):
        '''
        Class initializer
        -------------------------------------
        Input:
        X: input data
        y: label data
        -------------------------------------
        '''
        self.X = X
        self.N, self.D = X.shape

        self.y = y
        self.classes = np.unique(y)
        self.K = len(self.classes)
        # turn labels into one-hot-coding
        self.t_one_hot = np.zeros((self.N, len(self.classes)))
        self.t_one_hot[np.arange(self.N), self.y] = 1  # shape (N, K)

        self.max_iter = 50
        self.threshold = 1e-10

        # initialize weights
        self.W = 0.001 * np.random.random((self.D, self.K))  # shape (D, K)

    # Softmax function
    def softmax(self, a):
        e = np.exp(a - np.max(a, axis=1).reshape((-1, 1)))
        e_total = np.sum(e, axis=1).reshape((-1, 1))
        return e / e_total

    def IRLS(self):
        '''
        Function to update the weights iteratively
        '''

        for i in range(self.max_iter):

            print("Iteration ", i)

            # activations
            a = np.dot(self.X, self.W)  # shape (N, K)
            # posterior
            p = self.softmax(a)  # shape (N, K)

            # cross-entropy (to be minimized)
            E = - np.sum(self.t_one_hot * np.log(p + 1e-6))

            W = np.zeros((self.D, self.K))
            for k in self.classes:
                R = np.diag(p[:, k] * (1 - p[:, k]))

                # update new weights
                z = a[:, k] - np.dot(np.linalg.pinv(R), (p[:, k] - self.t_one_hot[:, k]))
                H = np.dot(self.X.T, R).dot(self.X)
                W[:, k] = np.linalg.pinv(H).dot(self.X.T).dot(R).dot(z)

                print("Updated weights for class ", k)

            self.W = W

            new_p = self.softmax(np.dot(self.X, self.W))
            new_E = - np.sum(self.t_one_hot * np.log(new_p + 1e-6))

            # stop if converge
            if E - new_E < self.threshold:
                print("Converged!")
                break

    def calculate_error(self, X_test, y_test):
        y_prob = self.softmax(np.dot(X_test, self.W))
        y_pred = list(np.argmax(y_prob, axis=1))
        error = np.sum(y_pred != y_test) / float(len(y_test))
        return error


def logisticRegression(filename, num_splits, train_percent):
    '''
    Input:
        filename: boston / digits
        num_splits: number of 80-20 train-test splits for evaluation
        train_percent: vector containing percentages of training data to be used for training
    ------------------------------------------------------------
    Output:
        test set error rates for each training set percent
    '''
    data, label = load_data(filename)

    error_matrix = np.zeros((num_splits, len(train_percent)))

    for i in range(num_splits):

        print("Split #", i)

        # Split the dataset into 80-20 train-test sets
        X_train, y_train, X_test, y_test = train_test_split(data, label=label)

        for j, p in enumerate(train_percent):
            print("Training with %s percent of training data" % p)

            # subset the training set
            train_max_idx = int(np.floor(p / 100.00 * X_train.shape[0]))
            X_train_p = X_train.loc[:train_max_idx]
            y_train_p = y_train.loc[:train_max_idx]

            # fit Logistic Regression model
            log_reg = LogisticRegression(X_train_p, y_train_p)
            log_reg.IRLS()

            # calculate test error
            error_matrix[i, j] = log_reg.calculate_error(X_test, y_test)

    pickle_on = open("./pickle/%s_logreg_error_matrix.pickle" % filename, 'wb')
    pickle.dump(error_matrix, pickle_on)
    pickle_on.close()

    mean_error = np.mean(error_matrix, axis=0)
    std_error = np.std(error_matrix, axis=0, ddof=1)

    return mean_error, std_error


if __name__ == '__main__':
    # filename = sys.argv[1]
    # num_splits = int(sys.argv[2])
    filename = 'boston'
    num_splits = int(5)

    mean_error, std_error = logisticRegression(filename, num_splits, [10, 25, 50, 75, 100])

print("Mean test errors are %s , with standard errors % s" % (mean_error, std_error))
