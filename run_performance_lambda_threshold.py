#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Item_based_lambda_discriminant import ItemBasedLambdaDiscriminantRecommender

from Recommenders.Base.non_personalized import TopPop
from Recommenders.KNN.item_knn_CF import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alpha import P3alphaRecommender
from Recommenders.GraphBased.RP3beta import RP3betaRecommender


from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.Movielens_10m.Movielens10MReader import Movielens10MReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader


from data.DataSplitter import DataSplitter_Warm

import matplotlib.pyplot as plt

import numpy as np


if __name__ == '__main__':


    dataSplitter = DataSplitter_Warm(Movielens10MReader)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()

    #ICM = dataSplitter.get_split_for_specific_ICM("ICM_all")


    personalized_recommender = TopPop(URM_train)
    non_personalized_recommender = ItemKNNCFRecommender(URM_train)


    recommender = ItemBasedLambdaDiscriminantRecommender(URM_train, non_personalized_recommender, personalized_recommender)

    recommender.fit()

    user_lambda = recommender.get_lambda_values()

    np.sort(user_lambda)


    map_performance_over_lambda = []


    for lambda_threshold in user_lambda:

        recommender.set_lambda_threshold(lambda_threshold)

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)

        print("Lambda threshold is {}, results: {}".format(lambda_threshold, results_run))

        map_performance_over_lambda.append(results_run["map"])






    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')


    plt.xlabel('lambda threshold')
    plt.ylabel("MAP")
    plt.title("Recommender MAP for increasing lambda threshold")


    plt.plot(np.arange(len(map_performance_over_lambda)), map_performance_over_lambda, linewidth=3, label="CBF data",
             linestyle = ":")

    plt.legend()

    plt.savefig("plots/MAP_over_lambda_{}_{}".format(
        personalized_recommender.RECOMMENDER_NAME, non_personalized_recommender.RECOMMENDER_NAME))

    plt.close()

