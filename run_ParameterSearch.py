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







def runParameterSearch(URM_train, URM_validation, dataReader_class, logFilePath ="results/"):

    from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython

    from ParameterTuning.AbstractClassSearch import DictionaryKeys


    ##########################################################################################################

    recommender_class = Lambda_BPR_Cython

    parameterSearch = BayesianSearch(recommender_class, URM_validation)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["pseudoInv"] = [True]
    hyperparamethers_range_dictionary["epochs"] = [200]
    hyperparamethers_range_dictionary["rcond"] = list(np.arange(0.10, 0.3, 0.05))
    hyperparamethers_range_dictionary["low_ram"] = [False]
    #hyperparamethers_range_dictionary["k"] = list(range(5,260,10))
    hyperparamethers_range_dictionary["learning_rate"] = [0.01]
    hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
    hyperparamethers_range_dictionary["batch_size"] = [1]
    hyperparamethers_range_dictionary["initialize"] = ["zero", "random"]#, "one", "random"]

    logFile = open(logFilePath + "Lambda_BPR_Cython" + "_{}_BayesianSearch.txt".format(dataReader_class.DATASET_SUBFOLDER[:-1]), "a")

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"save_eval":False},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: [],
                             DictionaryKeys.FIT_KEYWORD_ARGS: {"URM_test": URM_validation, "validation_every_n":1,
                                                               "lower_validatons_allowed":1},
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



    best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelize=False, n_cases=0,
                                             folderPath = logFilePath, namePrefix = "Lambda_BPR_Cython_{}_best_parameters_lambda".format(dataReader_class.DATASET_SUBFOLDER[:-1]))

    logFile.write("best_parameters: {}".format(best_parameters))
    logFile.flush()
    logFile.close()

    pickle.dump(best_parameters, open(logFilePath + "Lambda_BPR_Cython" +
                                      "_{}_best_parameters".format(dataReader_class.DATASET_SUBFOLDER[:-1]), "wb"), protocol=pickle.HIGHEST_PROTOCOL)




import os

import traceback
import multiprocessing



def read_data_split_and_search(dataReader_class):



    dataSplitter = DataSplitter_Warm(dataReader_class)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()





    runParameterSearch(URM_train, URM_validation, dataReader_class)



if __name__ == '__main__':

    dataReader_class_list = [
        Movielens1MReader,
        #Movielens20MReader,
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


