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
from data.Movielens_1m.Movielens1MReader import Movielens1MReader
from data.Movielens_10m.Movielens10MReader import Movielens10MReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader


from data.DataSplitter import DataSplitter_Warm

import matplotlib.pyplot as plt

import numpy as np
import pickle, itertools


if __name__ == '__main__':

    dataReader_class = Movielens1MReader


    dataSplitter = DataSplitter_Warm(dataReader_class)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()

    #ICM = dataSplitter.get_split_for_specific_ICM("ICM_all")
    #
    # subsample_mask = np.random.choice([True, False], size=URM_train.shape[1], p=[0.2, 0.8])
    #
    # URM_train = URM_train[:,subsample_mask]
    # URM_validation = URM_validation[:,subsample_mask]
    # URM_test = URM_test[:,subsample_mask]



    non_personalized_recommender = TopPop(URM_train)
    personalized_recommender = ItemKNNCFRecommender(URM_train)


    non_personalized_recommender.fit()
    personalized_recommender.fit()
    hybrid_all = ItemBasedLambdaDiscriminantRecommender(URM_train,
                                                        non_personalized_recommender = non_personalized_recommender,
                                                        personalized_recommender = personalized_recommender,
                                                        URM_validation = URM_validation)

    hybrid_pers_only = ItemBasedLambdaDiscriminantRecommender(URM_train,
                                                        non_personalized_recommender = None,
                                                        personalized_recommender = personalized_recommender,
                                                        URM_validation = URM_validation)

    dataset_name = dataReader_class.DATASET_SUBFOLDER[:-1]

    optimal_params = pickle.load(open("results/Lambda_BPR_Cython" +
                                      "_{}_best_parameters".format(dataset_name), "rb"))



    namePrefix = "Lambda_BPR_Cython_{}_best_parameters".format(dataset_name)

    try:
        hybrid_all.loadModel("results/", namePrefix=namePrefix)
    except:
        hybrid_all.fit(**optimal_params)
        hybrid_all.saveModel("results/", namePrefix=namePrefix)

    hybrid_pers_only.loadModel("results/", namePrefix=namePrefix)


    user_lambda = hybrid_all.get_lambda_values()
    user_lambda = np.sort(user_lambda)


    map_performance_hybrid_all = []
    map_performance_pers_only = []
    x_tick = []




    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')


    ## Plot baseline
    results_run_non_pers = non_personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
    results_run_pers = personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
    print("Personalized result: {}".format(results_run_pers))
    print("Non personalized result: {}".format(results_run_non_pers))



    for lambda_threshold_index in range(100, len(user_lambda), 100):

        lambda_threshold = user_lambda[lambda_threshold_index]

        hybrid_all.set_lambda_threshold(lambda_threshold)
        hybrid_pers_only.set_lambda_threshold(lambda_threshold)

        # results_hybrid_all = hybrid_all.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        # print("Lambda threshold is {}, result hybrid all: {}".format(lambda_threshold, results_hybrid_all))

        users_to_filter = hybrid_all.get_lambda_values()<lambda_threshold
        users_to_filter = np.arange(0, len(user_lambda), dtype=np.int)[users_to_filter]

        results_pers_only = hybrid_pers_only.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_to_filter)
        print("Lambda threshold is {}, result pers only: {}".format(lambda_threshold, results_pers_only))

        #map_performance_hybrid_all.append(results_hybrid_all["map"])
        map_performance_pers_only.append(results_pers_only["map"])
        x_tick.append(lambda_threshold)




        plt.xlabel('lambda threshold')
        plt.ylabel("MAP")
        plt.title("Recommender MAP for increasing lambda threshold")

        marker_list = ['o', 's', '^', 'v', 'D']
        marker_iterator_local = itertools.cycle(marker_list)


        # plt.plot(x_tick, map_performance_hybrid_all, linewidth=3, label="Hybrid all",
        #          linestyle = "-", marker = marker_iterator_local.__next__())

        plt.plot(x_tick, map_performance_pers_only, linewidth=3, label="Pers only",
                 linestyle = "-", marker = marker_iterator_local.__next__())

        plt.plot(x_tick, np.ones_like(x_tick)*results_run_non_pers["map"], linewidth=3, label="Non pers",
                 linestyle = ":", marker = marker_iterator_local.__next__())

        plt.plot(x_tick, np.ones_like(x_tick)*results_run_pers["map"], linewidth=3, label="Pers",
                 linestyle = "--", marker = marker_iterator_local.__next__())

        plt.legend()

        plt.savefig("results/MAP_over_lambda_{}_{}_{}".format(dataset_name,
            personalized_recommender.RECOMMENDER_NAME, non_personalized_recommender.RECOMMENDER_NAME))

        plt.close()

