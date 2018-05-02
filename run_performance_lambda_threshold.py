#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/04/18

@author: Maurizio Ferrari Dacrema
"""

from Item_based_lambda_discriminant import ItemBasedLambdaDiscriminantRecommender

from Base.non_personalized import TopPop, Random
from KNN.item_knn_CF import ItemKNNCFRecommender
from GraphBased.P3alpha import P3alphaRecommender
from GraphBased.RP3beta import RP3betaRecommender


from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.Movielens_1m.Movielens1MReader import Movielens1MReader
from data.Movielens_10m.Movielens10MReader import Movielens10MReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader
from data.XingChallenge2016.XingChallenge2016Reader import XingChallenge2016Reader
from data.BookCrossing.BookCrossingReader import BookCrossingReader


from data.DataSplitter import DataSplitter_Warm

import matplotlib.pyplot as plt

import numpy as np
import pickle, itertools
import scipy.sparse as sps



def plot_lambda_profile_length(user_lambda, URM_train, dataset_name, lambda_range, mode):

    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('profile length')
    plt.ylabel("user lambda")
    plt.title("Profile length and corresponding lambda")


    URM_train = sps.csr_matrix(URM_train)
    profile_length = np.ediff1d(URM_train.indptr)

    profile_length_user_id = np.argsort(profile_length)

    plt.scatter(profile_length[profile_length_user_id], user_lambda[profile_length_user_id], s=0.5)

    plt.savefig("results/Profile_length_over_lambda_{}_{}_{}".format(dataset_name, lambda_range, mode))

    plt.close()


from Base.metrics import roc_auc, precision, recall, map, ndcg, rr


def plot_lambda_user_performance(user_lambda, personalized_recommender, URM_train, URM_test, dataset_name):

    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('MAP')
    plt.ylabel("user lambda")
    plt.title("User MAP and corresponding lambda")

    URM_train = sps.csr_matrix(URM_train)
    URM_test = sps.csr_matrix(URM_test)

    user_int_test = np.ediff1d(URM_test.indptr) >= 2

    user_map = np.ones(URM_test.shape[0])*100


    for test_user in range(URM_test.shape[0]):

        if not user_int_test[test_user]:
            continue

        # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

        # Being the URM CSR, the indices are the non-zero column indexes
        relevant_items = URM_test.indices[URM_test.indptr[test_user]:URM_test.indptr[test_user+1]]


        recommended_items = personalized_recommender.recommend(user_id=test_user, exclude_seen=False,
                                                    n=5, filterTopPop=False)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        # roc_auc_ += roc_auc(is_relevant)
        # precision_ += precision(is_relevant)
        # recall_ += recall(is_relevant, relevant_items)
        try:
            map_ = map(is_relevant, relevant_items)
            user_map[test_user] = map_
        except:
            pass
        # mrr_ += rr(is_relevant)
        # ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)



    user_map = user_map[user_int_test]
    user_lambda = user_lambda[user_int_test]

    user_map_user_id = np.argsort(user_map)


    plt.scatter(user_map[user_map_user_id], user_lambda[user_map_user_id], s=0.5)

    plt.savefig("results/User_MAP_over_lambda_{}".format(dataset_name))

    plt.close()


    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('MAP')
    plt.ylabel("user lambda")
    plt.title("User MAP and corresponding lambda")

    lambda_user_id = np.argsort(user_lambda)

    plt.scatter(user_lambda[lambda_user_id],user_map[lambda_user_id], s=0.5)

    plt.savefig("results/Lambda_over_user_map_{}".format(dataset_name))

    plt.close()



def plot_lambda_user_performance_on_train(user_lambda, personalized_recommender, URM_train, dataset_name):

    URM_test = URM_train.copy()

    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('user lambda')
    plt.ylabel("MAP")
    plt.title("User MAP and corresponding lambda")

    URM_train = sps.csr_matrix(URM_train)
    URM_test = sps.csr_matrix(URM_test)

    user_int_test = np.ediff1d(URM_test.indptr) >= 2

    user_map = np.ones(URM_test.shape[0])*100


    for test_user in range(URM_test.shape[0]):

        if not user_int_test[test_user]:
            continue

        # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

        # Being the URM CSR, the indices are the non-zero column indexes
        relevant_items = URM_test.indices[URM_test.indptr[test_user]:URM_test.indptr[test_user+1]]


        recommended_items = personalized_recommender.recommend(user_id=test_user, exclude_seen=False,
                                                    n=5, filterTopPop=False)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        # roc_auc_ += roc_auc(is_relevant)
        # precision_ += precision(is_relevant)
        # recall_ += recall(is_relevant, relevant_items)
        try:
            map_ = map(is_relevant, relevant_items)
            user_map[test_user] = map_
        except:
            pass
        # mrr_ += rr(is_relevant)
        # ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)



    user_map = user_map[user_int_test]
    user_lambda = user_lambda[user_int_test]

    user_map_user_id = np.argsort(user_map)


    plt.scatter(user_map[user_map_user_id], user_lambda[user_map_user_id], s=0.5)

    plt.savefig("results/User_MAP_over_lambda_on_train_{}".format(dataset_name))

    plt.close()


    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('user lambda')
    plt.ylabel("MAP")
    plt.title("User MAP and corresponding lambda")

    lambda_user_id = np.argsort(user_lambda)

    plt.scatter(user_lambda[lambda_user_id],user_map[lambda_user_id], s=0.5)

    plt.savefig("results/Lambda_over_user_map_on_train_{}".format(dataset_name))

    plt.close()




def plot_hybrid_performance(dataReader_class):

    use_lambda = True

    for mode in ["pinv", "transpose"]:

        for negative in [True, False]:

            plot_CF_performance_on_lambda_threshold(dataReader_class, use_lambda = use_lambda, mode = mode, negative = negative)

    # plot_hybrid_performance_inner(dataReader_class, use_lambda = True, mode="pinv", negative=False)
    # plot_hybrid_performance_inner(dataReader_class, use_lambda = True, mode="transpose", negative=False)
    #plot_hybrid_performance_inner(dataReader_class, use_lambda = False)


#
#
# def plot_hybrid_performance_inner(dataReader_class, use_lambda = True, mode = "pinv", negative = False):
#
#     #dataReader_class = Movielens10MReader
#
#     if dataReader_class is BookCrossingReader or dataReader_class is XingChallenge2016Reader:
#
#         split_path = "results/split/" + dataReader_class.DATASET_SUBFOLDER[:-1] + "_"
#
#         URM_train = sps.load_npz(split_path + "URM_train.npz")
#         URM_test = sps.load_npz(split_path + "URM_test.npz")
#         URM_validation = sps.load_npz(split_path + "URM_validation.npz")
#
#
#     else:
#
#         dataSplitter = DataSplitter_Warm(dataReader_class)
#
#         URM_train = dataSplitter.get_URM_train()
#         URM_validation = dataSplitter.get_URM_validation()
#         URM_test = dataSplitter.get_URM_test()
#
#
#     #ICM = dataSplitter.get_split_for_specific_ICM("ICM_all")
#     #
#     # subsample_mask = np.random.choice([True, False], size=URM_train.shape[1], p=[0.2, 0.8])
#     #
#     # URM_train = URM_train[:,subsample_mask]
#     # URM_validation = URM_validation[:,subsample_mask]
#     # URM_test = URM_test[:,subsample_mask]
#
#
#     non_personalized_recommender = TopPop(URM_train)
#     personalized_recommender = ItemKNNCFRecommender(URM_train)
#     random_recommender = Random(URM_train)
#
#
#     non_personalized_recommender.fit()
#     personalized_recommender.fit()
#     random_recommender.fit()
#
#     hybrid_all = ItemBasedLambdaDiscriminantRecommender(URM_train,
#                                                         non_personalized_recommender = non_personalized_recommender,
#                                                         personalized_recommender = personalized_recommender,
#                                                         URM_validation = URM_validation)
#
#
#     hybrid_pers_only = ItemBasedLambdaDiscriminantRecommender(URM_train,
#                                                         non_personalized_recommender = None,
#                                                         personalized_recommender = personalized_recommender,
#                                                         URM_validation = URM_validation)
#
#     dataset_name = dataReader_class.DATASET_SUBFOLDER[:-1]
#
#     if negative:
#         lambda_range = "negative"
#     else:
#         lambda_range = "positive"
#
#
#     optimal_params = pickle.load(open("results/lambda_BPR/Lambda_BPR_Cython" +
#                                       "_{}_{}_{}_best_parameters".format(mode, lambda_range, dataset_name), "rb"))
#
#
#
#     print("Using params: {}".format(optimal_params))
#
#     namePrefix = "Lambda_BPR_Cython_{}_{}_{}_best_model".format(mode, lambda_range, dataset_name)
#
#     #
#     # try:
#     #     hybrid_all.loadModel("results/", namePrefix=namePrefix)
#     # except:
#     #     hybrid_all.fit(**optimal_params)
#     #     hybrid_all.saveModel("results/", namePrefix=namePrefix)
#
#     hybrid_all.loadModel("results/lambda_BPR/", namePrefix=namePrefix)
#     hybrid_pers_only.loadModel("results/lambda_BPR/", namePrefix=namePrefix)
#
#     if use_lambda:
#         user_lambda = hybrid_all.get_lambda_values()
#     else:
#         user_lambda = np.ediff1d(URM_train.indptr)
#
#
#     plot_lambda_profile_length(user_lambda, URM_train, dataset_name, lambda_range, mode)
#
#
#
#     #plot_lambda_user_performance(user_lambda, personalized_recommender, URM_train, URM_test, dataset_name)
#     #plot_lambda_user_performance_on_train(user_lambda, personalized_recommender, URM_train, dataset_name)
#
#
#
#     hybrid_all.user_lambda = user_lambda.copy()
#     hybrid_pers_only.user_lambda = user_lambda.copy()
#
#     user_lambda = np.sort(user_lambda)
#
#
#
#     map_performance_hybrid_all = []
#     map_performance_pers_only = []
#     map_performance_pers_only_less = []
#     map_performance_non_pers_only_less = []
#     map_performance_non_pers_only_over = []
#     map_performance_random_less = []
#
#     map_performance_pers_only_less_train_subset = []
#     map_performance_non_pers_only_less_train_subset = []
#     map_performance_pers_only_train_subset = []
#
#     x_tick = []
#
#
#
#
#     # Turn interactive plotting off
#     plt.ioff()
#
#     # Ensure it works even on SSH
#     plt.switch_backend('agg')
#
#
#     ## Plot baseline
#     results_run_non_pers = non_personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
#     results_run_pers = personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
#     print("Personalized result: {}".format(results_run_pers))
#     print("Non personalized result: {}".format(results_run_non_pers))
#
#
#     lambda_threshold = user_lambda[0]
#
#
#     for lambda_threshold_index in range(0, len(user_lambda), 100):
#         #
#         # if lambda_threshold_index != 0 and not lambda_threshold/user_lambda[lambda_threshold_index] <= 0.90:
#         #     continue
#
#         lambda_threshold = user_lambda[lambda_threshold_index]
#         #
#         # if lambda_threshold >= 6:
#         #     break
#
#
#
#         hybrid_all.set_lambda_threshold(lambda_threshold)
#         hybrid_pers_only.set_lambda_threshold(lambda_threshold)
#
#         results_hybrid_all = hybrid_all.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
#         print("Lambda threshold is {}, result hybrid all: {}".format(lambda_threshold, results_hybrid_all))
#
#         users_with_lower_lambda = hybrid_all.get_lambda_values()<lambda_threshold
#         users_with_lower_lambda = np.arange(0, len(user_lambda), dtype=np.int)[users_with_lower_lambda]
#
#         results_pers_only = hybrid_pers_only.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_with_lower_lambda)
#         print("Lambda threshold is {}, result pers only: {}".format(lambda_threshold, results_pers_only))
#
#         non_pers_only_over = non_personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_with_lower_lambda)
#         print("Lambda threshold is {}, result non pers over: {}".format(lambda_threshold, non_pers_only_over))
#
#
#
#
#
#         users_with_higher_lambda = hybrid_all.get_lambda_values()>lambda_threshold
#         users_with_higher_lambda = np.arange(0, len(user_lambda), dtype=np.int)[users_with_higher_lambda]
#
#         if lambda_threshold_index == 0:
#             results_pers_only_less = {"map": 0.0}
#             results_non_pers_only_less = {"map": 0.0}
#             results_random_less = {"map": 0.0}
#         else:
#             results_pers_only_less = personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_with_higher_lambda)
#             print("Lambda threshold is {}, result pers only less than lambda: {}".format(lambda_threshold, results_pers_only_less))
#
#             results_non_pers_only_less = non_personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_with_higher_lambda)
#             print("Lambda threshold is {}, result non pers only less than lambda: {}".format(lambda_threshold, results_non_pers_only_less))
#
#             results_random_less = random_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True, filterCustomUsers=users_with_higher_lambda)
#             print("Lambda threshold is {}, result random less than lambda: {}".format(lambda_threshold, results_random_less))
#
#
#         if lambda_threshold_index == 0:
#             results_pers_only_less_train_subset = {"map": 0.0}
#             results_non_pers_only_less_train_subset = {"map": 0.0}
#         else:
#             URM_train_subset = URM_train[users_with_lower_lambda,:]
#             URM_test_subset = URM_test[users_with_lower_lambda,:]
#             personalized_recommender_train_subset = ItemKNNCFRecommender(URM_train_subset)
#             personalized_recommender_train_subset.fit()
#             results_pers_only_less_train_subset = personalized_recommender_train_subset.evaluateRecommendations(URM_test_subset, at=5, exclude_seen=True)
#             print("Lambda threshold is {}, result pers only less than lambda_train_subset: {}".format(lambda_threshold, results_pers_only_less))
#
#             non_personalized_recommender_train_subset = TopPop(URM_train_subset)
#             non_personalized_recommender_train_subset.fit()
#             results_non_pers_only_less_train_subset = non_personalized_recommender_train_subset.evaluateRecommendations(URM_test_subset, at=5, exclude_seen=True)
#             print("Lambda threshold is {}, result non pers only less than lambda_train_subset: {}".format(lambda_threshold, results_non_pers_only_less))
#
#
#         URM_train_subset = URM_train[users_with_higher_lambda,:]
#         URM_test_subset = URM_test[users_with_higher_lambda,:]
#         personalized_recommender_train_subset = ItemKNNCFRecommender(URM_train_subset)
#         personalized_recommender_train_subset.fit()
#         results_pers_only_train_subset = personalized_recommender_train_subset.evaluateRecommendations(URM_test_subset, at=5, exclude_seen=True)
#         print("Lambda threshold is {}, result pers only: {}".format(lambda_threshold, results_pers_only_train_subset))
#
#
#         map_performance_hybrid_all.append(results_hybrid_all["map"])
#         map_performance_pers_only.append(results_pers_only["map"])
#         map_performance_pers_only_less.append(results_pers_only_less["map"])
#         map_performance_non_pers_only_less.append(results_non_pers_only_less["map"])
#         map_performance_non_pers_only_over.append(non_pers_only_over["map"])
#
#
#         map_performance_random_less.append(results_random_less["map"])
#
#         map_performance_pers_only_less_train_subset.append(results_pers_only_less_train_subset["map"])
#         map_performance_non_pers_only_less_train_subset.append(results_non_pers_only_less_train_subset["map"])
#         map_performance_pers_only_train_subset.append(results_pers_only_train_subset["map"])
#         x_tick.append(lambda_threshold)
#
#         # Turn interactive plotting off
#         plt.ioff()
#
#         # Ensure it works even on SSH
#         plt.switch_backend('agg')
#
#
#         plt.xlabel('lambda threshold')
#         plt.ylabel("MAP")
#         plt.title("Recommender MAP for increasing lambda threshold")
#
#         marker_list = ['o', 's', '^', 'v', 'D']
#         marker_iterator_local = itertools.cycle(marker_list)
#
#
#         # plt.plot(x_tick, map_performance_hybrid_all, linewidth=3, label="Hybrid all",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#
#         plt.plot(x_tick, map_performance_pers_only, linewidth=3, label="Item KNN CF, lambda > threshold",
#                  linestyle = "-", marker = marker_iterator_local.__next__())
#
#         plt.plot(x_tick, map_performance_pers_only_less, linewidth=3, label="Item KNN CF, lambda < threshold",
#                  linestyle = "-", marker = marker_iterator_local.__next__())
#
#         plt.plot(x_tick, map_performance_non_pers_only_less, linewidth=3, label="TopPop, lambda < threshold",
#                  linestyle = "-", marker = marker_iterator_local.__next__())
#
#         plt.plot(x_tick, map_performance_non_pers_only_over, linewidth=3, label="TopPop, lambda > threshold",
#                  linestyle = "-", marker = marker_iterator_local.__next__())
#
#         # plt.plot(x_tick, map_performance_random_less, linewidth=3, label="Random less lambda",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#
#
#         # plt.plot(x_tick, np.ones_like(x_tick)*results_run_non_pers["map"], linewidth=3, label="TopPop",
#         #          linestyle = ":", marker = marker_iterator_local.__next__())
#
#         # plt.plot(x_tick, np.ones_like(x_tick)*results_run_pers["map"], linewidth=3, label="Item KNN CF",
#         #          linestyle = "--", marker = marker_iterator_local.__next__())
#
#         legend = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)
#         legend = [legend]
#
#         if use_lambda:
#             discrminant_is = "_lambda"
#         else:
#             discrminant_is = "_profile_len"
#
#         plt.savefig("results/MAP_over_lambda_{}_{}_{}_train_on_all_URM{}".format(dataset_name,
#             personalized_recommender.RECOMMENDER_NAME, non_personalized_recommender.RECOMMENDER_NAME, discrminant_is),
#             additional_artists=legend, bbox_inches="tight")
#
#         plt.close()
#
#
#
#
#         #
#         #
#         #
#         # # Turn interactive plotting off
#         # plt.ioff()
#         #
#         # # Ensure it works even on SSH
#         # plt.switch_backend('agg')
#         #
#         #
#         # plt.xlabel('lambda threshold')
#         # plt.ylabel("MAP")
#         # plt.title("Recommender MAP for increasing lambda threshold")
#         #
#         # marker_list = ['o', 's', '^', 'v', 'D']
#         # marker_iterator_local = itertools.cycle(marker_list)
#         #
#         #
#         # plt.plot(x_tick, map_performance_hybrid_all, linewidth=3, label="Hybrid all",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#         #
#         # plt.plot(x_tick, map_performance_pers_only_train_subset, linewidth=3, label="Personalized only",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#         #
#         # plt.plot(x_tick, map_performance_pers_only_less_train_subset, linewidth=3, label="Personalized only less lambda",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#         #
#         # plt.plot(x_tick, map_performance_non_pers_only_less_train_subset, linewidth=3, label="Non personalized only less lambda",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#         #
#         # plt.plot(x_tick, map_performance_random_less, linewidth=3, label="Random less lambda",
#         #          linestyle = "-", marker = marker_iterator_local.__next__())
#         #
#         #
#         # plt.plot(x_tick, np.ones_like(x_tick)*results_run_non_pers["map"], linewidth=3, label="TopPop",
#         #          linestyle = ":", marker = marker_iterator_local.__next__())
#         #
#         # plt.plot(x_tick, np.ones_like(x_tick)*results_run_pers["map"], linewidth=3, label="Item KNN Collaborative",
#         #          linestyle = "--", marker = marker_iterator_local.__next__())
#         #
#         # legend = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
#         # legend = [legend]
#         #
#         # plt.savefig("results/MAP_over_lambda_{}_{}_{}_train_on_subset{}".format(dataset_name,
#         #     personalized_recommender.RECOMMENDER_NAME, non_personalized_recommender.RECOMMENDER_NAME, discrminant_is),
#         #     additional_artists=legend, bbox_inches="tight")
#         #
#         # plt.close()
#




def plot_CF_performance_on_lambda_threshold(dataReader_class, use_lambda = True, mode = "pinv", negative = False):


    if dataReader_class is BookCrossingReader or dataReader_class is XingChallenge2016Reader:

        split_path = "results/split/" + dataReader_class.DATASET_SUBFOLDER[:-1] + "_"

        URM_train = sps.load_npz(split_path + "URM_train.npz")
        URM_test = sps.load_npz(split_path + "URM_test.npz")
        URM_validation = sps.load_npz(split_path + "URM_validation.npz")


    else:

        dataSplitter = DataSplitter_Warm(dataReader_class)

        URM_train = dataSplitter.get_URM_train()
        URM_validation = dataSplitter.get_URM_validation()
        URM_test = dataSplitter.get_URM_test()



    non_personalized_recommender = TopPop(URM_train)
    personalized_recommender = ItemKNNCFRecommender(URM_train)


    non_personalized_recommender.fit()
    personalized_recommender.fit()



    dataset_name = dataReader_class.DATASET_SUBFOLDER[:-1]

    if negative:
        lambda_range = "negative"
    else:
        lambda_range = "positive"


    optimal_params = pickle.load(open("results/lambda_BPR/Lambda_BPR_Cython" +
                                      "_{}_{}_{}_best_parameters".format(mode, lambda_range, dataset_name), "rb"))



    print("Using params: {}".format(optimal_params))

    namePrefix = "Lambda_BPR_Cython_{}_{}_{}_best_model.npz".format(mode, lambda_range, dataset_name)

    npzfile = np.load("results/lambda_BPR/" + namePrefix)
    user_lambda = npzfile["user_lambda"]

    plot_lambda_profile_length(user_lambda, URM_train, dataset_name, lambda_range, mode)



    #plot_lambda_user_performance(user_lambda, personalized_recommender, URM_train, URM_test, dataset_name)
    #plot_lambda_user_performance_on_train(user_lambda, personalized_recommender, URM_train, dataset_name)


    user_lambda_sorted = np.sort(user_lambda)

    map_performance_TopPop = []
    map_performance_CF = []
    map_performance_SLIM_lambda = []

    x_tick = []




    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')


    ## Plot baseline
    # results_run_non_pers = non_personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
    # results_run_pers = personalized_recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
    # print("Personalized result: {}".format(results_run_pers))
    # print("Non personalized result: {}".format(results_run_non_pers))


    lambda_step = int(URM_train.shape[0] * 0.10)


    for lambda_threshold_index in range(0, len(user_lambda_sorted)-lambda_step, lambda_step):

        lambda_threshold_min = user_lambda_sorted[lambda_threshold_index]
        lambda_threshold_max = user_lambda_sorted[lambda_threshold_index+lambda_step]

        #
        # if lambda_threshold >= 6:
        #     break

        users_involved_mask = np.logical_and(user_lambda <= lambda_threshold_max, user_lambda >= lambda_threshold_min)
        users_involved = np.arange(0, len(user_lambda), dtype=np.int)[users_involved_mask]
        users_not_involved = np.arange(0, len(user_lambda), dtype=np.int)[np.logical_not(users_involved_mask)]


        URM_train_current_user_batch = URM_train[users_involved,:]
        URM_test_current_user_batch = URM_test[users_involved,:]


        non_personalized_recommender = TopPop(URM_train_current_user_batch)
        personalized_recommender = ItemKNNCFRecommender(URM_train_current_user_batch)


        non_personalized_recommender.fit()
        personalized_recommender.fit()



        results_personalized_CF = personalized_recommender.evaluateRecommendations(URM_test_current_user_batch, at=5, exclude_seen=True,
                                                                    filterCustomUsers=users_not_involved)

        print("Lambda threshold is {}-{}, result personalized: {}".format(lambda_threshold_min, lambda_threshold_max, results_personalized_CF))

        results_non_personalized = non_personalized_recommender.evaluateRecommendations(URM_test_current_user_batch, at=5, exclude_seen=True,
                                                                    filterCustomUsers=users_not_involved)

        print("Lambda threshold is {}-{}, result non personalized: {}".format(lambda_threshold_min, lambda_threshold_max, results_non_personalized))



        map_performance_TopPop.append(results_non_personalized["map"])
        map_performance_CF.append(results_personalized_CF["map"])
        #map_performance_SLIM_lambda.append(results_pers_only["map"])


        x_tick.append(lambda_threshold_min)

        # Turn interactive plotting off
        plt.ioff()

        # Ensure it works even on SSH
        plt.switch_backend('agg')


        plt.xlabel('lambda threshold')
        plt.ylabel("MAP")
        plt.title("Recommender MAP for increasing lambda threshold")

        marker_list = ['o', 's', '^', 'v', 'D']
        marker_iterator_local = itertools.cycle(marker_list)


        # plt.plot(x_tick, map_performance_hybrid_all, linewidth=3, label="Hybrid all",
        #          linestyle = "-", marker = marker_iterator_local.__next__())
        #
        # width = 0.05
        #
        # plt.bar(np.array(x_tick), map_performance_TopPop, width = width, label="TopPop")
        #
        # plt.bar(np.array(x_tick) + 0.1, map_performance_CF, width = width, label="Item KNN CF")

        plt.plot(x_tick, map_performance_TopPop, linewidth=3, label="TopPop",
                 linestyle = "-", marker = marker_iterator_local.__next__())

        plt.plot(x_tick, map_performance_CF, linewidth=3, label="Item KNN CF",
                 linestyle = "-", marker = marker_iterator_local.__next__())



        legend = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)
        legend = [legend]

        if use_lambda:
            discrminant_is = "lambda"
        else:
            discrminant_is = "profile_len"

        plt.savefig("results/plot/MAP_over_lambda_{}_mode_{}_range_{}__{}_{}_discrminant_is_{}".format(dataset_name, mode, lambda_range,
            personalized_recommender.RECOMMENDER_NAME, non_personalized_recommender.RECOMMENDER_NAME, discrminant_is),
            additional_artists=legend, bbox_inches="tight")

        plt.close()





import multiprocessing, traceback

if __name__ == '__main__':

    dataReader_class_list = [
        Movielens1MReader,
        Movielens10MReader,
        #NetflixEnhancedReader,
        #BookCrossingReader,
        #XingChallenge2016Reader
    ]


    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
    # resultList = pool.map(plot_hybrid_performance, dataReader_class_list)


    for dataReader_class in dataReader_class_list:
        try:
            plot_hybrid_performance(dataReader_class)
        except Exception as e:

            print("On recommender {} Exception {}".format(dataReader_class, str(e)))
            traceback.print_exc()


