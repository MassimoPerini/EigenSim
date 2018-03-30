
import os
from loader.Dataset import Dataset
from loader.Dataset import load_dataframe, clean_dataframe, save_dataframe
#from Recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython as SLIM_CV
#from Recommenders.SLIM_BPR_hold_on.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython as SLIM_HO
from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython as Lambda_HO
#from Recommenders.Lambda_KFold.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython as LKF
import multiprocessing
import itertools
import numpy as np
from scipy.sparse import *
import scipy as sps

#LOAD DATASET

#file_path = os.path.expanduser('dataset/movielens_1M/ratings_cleaned.csv')
file_path = os.path.expanduser('dataset/1M.csv')
dataframe = load_dataframe(file_path, ',')
#dataset = Dataset(dataframe, user_key='userId', item_key='movieId', rating_key='rating')
dataset = Dataset(dataframe, user_key='userId', item_key='movieId', rating_key='rating') #"User-ID";"ISBN";"Book-Rating"
matrix = dataset.get_all_dataset()

#split train test

trainPercentage = 0.8
URM_all = matrix

URM_all = URM_all.tocoo()
numInteractions = len(URM_all.data)

#hold out

mask = np.random.choice([True, False], numInteractions, p=[trainPercentage, 1 - trainPercentage])
URM_train = coo_matrix((URM_all.data[mask], (URM_all.row[mask], URM_all.col[mask])), shape=URM_all.shape)
URM_train = URM_train.tocsr()
mask = np.logical_not(mask)
URM_test = coo_matrix((URM_all.data[mask], (URM_all.row[mask], URM_all.col[mask])), shape=URM_all.shape)
URM_test = URM_test.tocsr()

print("-----LAMBDA-------\nMatrix: rows", URM_all.shape[0]," cols: ", URM_all.shape[1], "items: ", URM_all.nnz)

print(URM_train.shape)

#LAMBDA START


lam = Lambda_HO(URM_train, recompile_cython=True, sgd_mode="adagrad", pseudoInv=True, rcond = 0.13, check_stability=False, save_lambda=True, save_eval=True)
#lam.fit(epochs=400,URM_test=URM_test ,sgd_mode="adagrad", learning_rate=0.00015, alpha=0.001, batch_size=1, validate_every_N_epochs=1)#229961 100k

lam.fit(epochs=12,URM_test=URM_test, learning_rate=0.0005, alpha=0, batch_size=1, validate_every_N_epochs=1, start_validation_after_N_epochs=0, initialize = "zero")
#res = lam.evaluateRecommendations(URM_test, at=5, check_stability=True)