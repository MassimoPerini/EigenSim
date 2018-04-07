#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender import Recommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender

from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython
import numpy as np


class ItemBasedLambdaDiscriminantRecommender(Similarity_Matrix_Recommender, Recommender):
    """
    This recommender uses the learned lambda to determine whether to use a personalized or non personalized recommender for a given user
    If the user's lambda is higher than lambda_threshold, a personalized recommender is used, otherwise a non personalized one
    """

    RECOMMENDER_NAME = "ItemBasedLambdaDiscriminantRecommender"


    def __init__(self, URM_train, non_personalized_recommender, personalized_recommender, URM_validation = None):
        super(ItemBasedLambdaDiscriminantRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.non_personalized_recommender = non_personalized_recommender
        self.personalized_recommender = personalized_recommender

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None



    def fit(self):

        pseudoinverse_size = self.URM_train.shape[0] * self.URM_train.shape[1]*32

        if pseudoinverse_size >= 3*1e+9:
            input("Pseudoinverse size is: {:.2f} GB. Continue?".format(pseudoinverse_size/1e+9))


        lambda_cython = Lambda_BPR_Cython(self.URM_train, recompile_cython=False, sgd_mode="adagrad",
                                          pseudoInv=True, check_stability=False,
                                          save_lambda=False, save_eval=False)


        lambda_cython.fit(epochs=2, URM_test = self.URM_validation, learning_rate=0.0005, alpha=0,
                          batch_size=1, validate_every_N_epochs=1,
                          start_validation_after_N_epochs=0, initialize = "zero", rcond = 0.13)

        self.user_lambda = lambda_cython.get_lambda()



    def get_lambda_values(self):

        return self.user_lambda.copy()


    def set_lambda_threshold(self, lambda_threshold = 0.0):

        self.lambda_threshold = lambda_threshold




    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        if self.user_lambda[user_id] >= self.lambda_threshold:
            return self.personalized_recommender.recommend(user_id, n=n, exclude_seen=exclude_seen,
                                                           filterTopPop = filterTopPop, filterCustomItems = filterCustomItems)

        else:
            return self.non_personalized_recommender.recommend(user_id, n=n, exclude_seen=exclude_seen,
                                                               filterTopPop = filterTopPop, filterCustomItems = filterCustomItems)








    def saveModel(self, folderPath, namePrefix = None):


        print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        np.savez(folderPath + "{}.npz".format(namePrefix), user_lambda = self.user_lambda)




    def loadModel(self, folderPath, namePrefix = None):

        print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"


        npzfile = np.load(folderPath + "{}.npz".format(namePrefix))

        for attrib_name in npzfile.files:
             self.__setattr__(attrib_name, npzfile[attrib_name])



