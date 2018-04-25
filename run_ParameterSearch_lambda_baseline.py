#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.Base.non_personalized import TopPop
from Recommenders.KNN.item_knn_CF import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alpha import P3alphaRecommender
from Recommenders.GraphBased.RP3beta import RP3betaRecommender


from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader
from data.Movielens_1m.Movielens1MReader import Movielens1MReader
from data.Movielens_10m.Movielens10MReader import Movielens10MReader
from data.Movielens_20m.Movielens20MReader import Movielens20MReader
from data.BookCrossing.BookCrossingReader import BookCrossingReader


from ParameterTuning.BayesianSearch import BayesianSearch

from data.DataSplitter import DataSplitter_Warm

import numpy as np
import scipy.sparse as sps
import pickle







def runParameterSearch(URM_train, URM_validation, URM_test, dataReader_class, logFilePath ="results/"):

    from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython

    from ParameterTuning.AbstractClassSearch import DictionaryKeys

    from Recommenders.KNN.user_knn_CF import UserKNNCFRecommender
    from Recommenders.KNN.item_knn_CF import ItemKNNCFRecommender
    from Recommenders.GraphBased.P3alpha import P3alphaRecommender
    from Recommenders.SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_BPR_Cython, SLIM_Structure_MSE_Cython
    from Recommenders.MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
    from Recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython




    ##########################################################################################################
    #
    # from ParameterTuning.AbstractClassSearch import evaluation_function_default
    #
    #
    #
    # recommender = TopPop(URM_train)
    #
    # recommender.fit()
    #
    # result_baseline = evaluation_function_default(recommender, URM_validation, {})
    #
    # output_root_path = logFilePath + TopPop.RECOMMENDER_NAME + "_{}_BayesianSearch.txt".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # output_file = open(output_root_path, "a")
    #
    # print("ParameterSearch: Best result evaluated on URM_test. Results: {}\n".format(result_baseline))
    # output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}\n".format(result_baseline))
    # output_file.close()


    ##########################################################################################################
    #
    # recommender_class = Lambda_BPR_Cython
    #
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["pseudoInv"] = [True]
    # hyperparamethers_range_dictionary["epochs"] = [10]
    # hyperparamethers_range_dictionary["rcond"] = list(np.arange(0.10, 0.3, 0.02))
    # hyperparamethers_range_dictionary["low_ram"] = [False]
    # #hyperparamethers_range_dictionary["k"] = list(range(5,260,10))
    # hyperparamethers_range_dictionary["learning_rate"] = [0.01]
    # hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
    # hyperparamethers_range_dictionary["batch_size"] = [1]
    # hyperparamethers_range_dictionary["initialize"] = ["zero", "random"]#, "one", "random"]
    #
    # output_root_path = logFilePath + "Lambda_BPR_Cython" + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"save_eval":False},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: [],
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"URM_test": URM_validation, "validation_every_n":1,
    #                                                            "lower_validatons_allowed":5},
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=10)

    #
    #
    #
    #
    # recommender_class = UserKNNCFRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    # hyperparamethers_range_dictionary["similarity"] = ['cosine', 'pearson', 'adjusted', 'jaccard']
    # hyperparamethers_range_dictionary["normalize"] = [True, False]
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=10)
    #
    # parameterSearch.evaluate_on_test(URM_test)


    ##########################################################################################################
    #
    # recommender_class = ItemKNNCFRecommender
    #
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    # hyperparamethers_range_dictionary["similarity"] = ['cosine', 'pearson', 'adjusted', 'jaccard']
    # hyperparamethers_range_dictionary["normalize"] = [True, False]
    #
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=10)
    #
    # parameterSearch.evaluate_on_test(URM_test)
    #



    # ##########################################################################################################
    #
    #
    # recommender_class = MultiThreadSLIM_RMSE
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100]
    # hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
    # hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
    #
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelize=False)
    #
    # logFile.write("best_parameters: {}".format(best_parameters))
    # logFile.flush()
    # logFile.close()
    #
    # pickle.dump(best_parameters, open(logFilePath + recommender_class.RECOMMENDER_NAME + "_best_parameters", "wb"), protocol=pickle.HIGHEST_PROTOCOL)



   ##########################################################################################################
    #
    # recommender_class = P3alphaRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["alpha"] = list(np.arange(0.1, 2.1, 0.2))
    # hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]
    #
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=10)
    #
    # parameterSearch.evaluate_on_test(URM_test)


    ##########################################################################################################
    #
    # recommender_class = RP3betaRecommender
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [50, 100, 150, 200]
    # hyperparamethers_range_dictionary["alpha"] = list(np.arange(0.1, 1.7, 0.2))
    # hyperparamethers_range_dictionary["beta"] = list(np.arange(0.1, 1.7, 0.2))
    # hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]
    #
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_BayesianSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 6)
    #
    # logFile.write("best_parameters: {}".format(best_parameters))
    # logFile.flush()
    # logFile.close()
    #
    # pickle.dump(best_parameters, open(logFilePath + recommender_class.RECOMMENDER_NAME + "_best_parameters", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    ##########################################################################################################
    #
    # recommender_class = FunkSVD
    # parameterSearch = GridSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30]
    # hyperparamethers_range_dictionary["epochs"] = [5, 10, 20]
    # hyperparamethers_range_dictionary["reg"] = [1e-2, 1e-3, 1e-4, 1e-5]
    # hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
    #
    # logFile = open(logFilePath + recommender_class.RECOMMENDER_NAME + "_GridSearch.txt", "a")
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelPoolSize = 8)
    #
    # logFile.write("best_parameters: {}".format(best_parameters))
    # logFile.flush()
    # logFile.close()
    #
    # pickle.dump(best_parameters, open(logFilePath + recommender_class.RECOMMENDER_NAME + "_best_parameters", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    #
    # ##########################################################################################################

    recommender_class = MF_BPR_Cython
    parameterSearch = BayesianSearch(recommender_class, URM_validation)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["num_factors"] = [5, 10, 30, 50, 100]
    hyperparamethers_range_dictionary["epochs"] = [500]
    hyperparamethers_range_dictionary["batch_size"] = [1]
    hyperparamethers_range_dictionary["user_reg"] = [0.0]#, 1e-2, 1e-3, 1e-4, 1e-5]
    hyperparamethers_range_dictionary["positive_reg"] = [0.0]#, 1e-2, 1e-3, 1e-4, 1e-5]
    hyperparamethers_range_dictionary["negative_reg"] = [0.0]#, 1e-2, 1e-3, 1e-4, 1e-5]
    hyperparamethers_range_dictionary["learning_rate"] = [0.005]
    hyperparamethers_range_dictionary["sgd_mode"] = ["adam"]

    output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"positive_threshold": 3, "URM_validation": URM_validation},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":2, "stop_on_validation":True, "lower_validatons_allowed":30},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


    best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=30)

    parameterSearch.evaluate_on_test(URM_test)


    #########################################################################################################
    #
    # recommender_class = SLIM_BPR_Cython
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["sgd_mode"] = ["adam"]
    # hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-3, 1e-6, 1e-9]
    # hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-3, 1e-6, 1e-9]
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':True, 'positive_threshold':0,
    #                                                                    "URM_validation": URM_validation.copy()},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True, "lower_validatons_allowed":10},
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=30)
    #
    # parameterSearch.evaluate_on_test(URM_test)



    #########################################################################################################
    #
    # recommender_class = SLIM_Structure_BPR_Cython
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
    # hyperparamethers_range_dictionary["lambda_1"] = [0.0, 1e-3, 1e-6, 1e-9]
    # hyperparamethers_range_dictionary["lambda_2"] = [0.0, 1e-3, 1e-6, 1e-9]
    #
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_validation},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True, "lower_validatons_allowed":10},
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=30)
    #
    # parameterSearch.evaluate_on_test(URM_test)


    ##########################################################################################################
    #
    # recommender_class = SLIM_Structure_MSE_Cython
    # parameterSearch = BayesianSearch(recommender_class, URM_validation)
    #
    # hyperparamethers_range_dictionary = {}
    # hyperparamethers_range_dictionary["topK"] = [100, 200, 500]
    # hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
    # hyperparamethers_range_dictionary["lambda_1"] = [0.0, 1e-3, 1e-6, 1e-9]
    # hyperparamethers_range_dictionary["lambda_2"] = [0.0, 1e-3, 1e-6, 1e-9]
    #
    #
    # output_root_path = logFilePath + recommender_class.RECOMMENDER_NAME + "_{}".format(dataReader_class.DATASET_SUBFOLDER[:-1])
    #
    # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
    #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"URM_validation": URM_validation},
    #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
    #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True, "lower_validatons_allowed":10},
    #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
    #
    #
    # best_parameters = parameterSearch.search(recommenderDictionary, output_root_path = output_root_path, parallelize=False, n_cases=30)
    #
    # parameterSearch.evaluate_on_test(URM_test)

    ##########################################################################################################







import os

import traceback
import multiprocessing



def read_data_split_and_search(dataReader_class):



    dataSplitter = DataSplitter_Warm(dataReader_class)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()





    runParameterSearch(URM_train, URM_validation, URM_test, dataReader_class)



if __name__ == '__main__':

    dataReader_class_list = [
        Movielens1MReader,
        Movielens10MReader,
        #NetflixEnhancedReader,
        #BookCrossingReader,
        #XingChallenge2016Reader
    ]


    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
    # resultList = pool.map(read_data_split_and_search, dataReader_class_list)


    for dataReader_class in dataReader_class_list:
        try:
            read_data_split_and_search(dataReader_class)
        except Exception as e:

            print("On recommender {} Exception {}".format(dataReader_class, str(e)))
            traceback.print_exc()


