# data exploration and testing

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

gscores = pd.read_csv('ml-20m/genome-scores.csv')
gtags = pd.read_csv('ml-20m/genome-tags.csv')
links = pd.read_csv('ml-20m/links.csv')
movies = pd.read_csv('ml-20m/movies.csv')
ratings = pd.read_csv('ml-20m/ratings.csv')
tags = pd.read_csv('ml-20m/tags.csv')

# gscores.shape  #(11709768, 3)
# gtags.shape  #(1128, 2)
# links.shape  #(27278, 3)
# movies.shape  #(27278, 3)
# ratings.shape  #(20000263, 4)
# tags.shape  #(465564, 4)

tags.columns.tolist()
tags['tag'].nunique()  # 38643
tags['movieId'].nunique()  # 19545
# average of 1.97 movies per tag

# genre - get_dummies


def dummy_dict(x, sep='|', value=1):
    x1 = x.split(sep)
    return dict.fromkeys(x1, value)


genres = movies['genres'].apply(dummy_dict)
genres = pd.concat([movies[['movieId', 'title']], pd.DataFrame(genres.tolist()).fillna(0)], axis=1)

# genres.shape  # (27278, 22)

# Matrix Factorization


class NmfL2:

    def __init__(self, R, k, alpha, beta, iter):
        """
        Class to perform matrix factorization with L2 regularization for multiple choices of k
        :param R: Edgelist of interactions [user, item, value]
        :param k: array of values of k to choose
        :param alpha: learning rate
        :param beta: regularization parameter
        :param iter: number of iterations for each
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.kn = 0
        self.alpha = alpha
        self.beta = beta
        self.iter = iter

    def train(self):
        # Initialize matrix
        self.P = np.random.normal(scale=1./self.k[self.kn], size=(self.num_users, self.k[self.kn]))
        self.Q = np.random.normal(scale=1./self.k[self.kn], size=(self.num_items, self.k[self.kn]))

        # Initialize embeddings
        self.p = np.zeros(self.num_users)
        self.q = np.zeros(self.num_items)
        self.r = np.mean(self.R[np.where(self.R != 0)])


kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

for train_index, test_index in kf.split(X):
      print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]