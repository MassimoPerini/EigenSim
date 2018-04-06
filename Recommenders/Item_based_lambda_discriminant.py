#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender import Recommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender

from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython



class ItemBasedLambdaDiscriminantRecommender(Similarity_Matrix_Recommender, Recommender):
    """
    This recommender uses the learned lambda to determine whether to use a personalized or non personalized recommender for a given user
    If the user's lambda is higher than lambda_threshold, a personalized recommender is used, otherwise a non personalized one
    """

    RECOMMENDER_NAME = "ItemBasedLambdaDiscriminantRecommender"


    def __init__(self, URM_train, non_personalized_recommender, personalized_recommender):
        super(ItemBasedLambdaDiscriminantRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train, 'csr')
        self.non_personalized_recommender = non_personalized_recommender
        self.personalized_recommender = personalized_recommender



    def fit(self, URM_validation = None):

        self.non_personalized_recommender.fit()
        self.personalized_recommender.fit()


        lambda_cython = Lambda_BPR_Cython(self.URM_train, recompile_cython=True, sgd_mode="adagrad", pseudoInv=True, rcond = 0.14, check_stability=False, save_lambda=False, save_eval=False)

        lambda_cython.fit(epochs=12, URM_test=URM_validation, learning_rate=0.003, alpha=0, batch_size=1, validate_every_N_epochs=1, start_validation_after_N_epochs=0, initialize = "zero")

        self.user_lambda = lambda_cython.get_lambda()



    def get_lambda_values(self):

        return self.user_lambda.copy()


    def set_lambda_threshold(self, lambda_threshold = 0.0):

        self.lambda_threshold = lambda_threshold




    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        if self.user_lambda[user_id] >= self.lambda_threshold:
            self.personalized_recommender.recommend(user_id, n=n, exclude_seen=exclude_seen, filterTopPop = filterTopPop, filterCustomItems = filterCustomItems)

        else:
            self.non_personalized_recommender.recommend(user_id, n=n, exclude_seen=exclude_seen, filterTopPop = filterTopPop, filterCustomItems = filterCustomItems)

