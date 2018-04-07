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


from ParameterTuning.BayesianSearch import BayesianSearch

from data.DataSplitter import DataSplitter_Warm

import numpy as np
import scipy.sparse as sps
import pickle







def runParameterSearch(URM_train, URM_validation, logFilePath ="results/"):

    from Recommenders.Lambda.Cython.Lambda_BPR_Cython import Lambda_BPR_Cython

    from ParameterTuning.AbstractClassSearch import DictionaryKeys


    ##########################################################################################################

    recommender_class = Lambda_BPR_Cython

    parameterSearch = BayesianSearch(recommender_class, URM_validation)

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["pseudoInv"] = [True]
    hyperparamethers_range_dictionary["epochs"] = [50]
    hyperparamethers_range_dictionary["rcond"] = list(np.arange(0.05, 0.5, 0.02))
    hyperparamethers_range_dictionary["learning_rate"] = [0.01]
    hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
    hyperparamethers_range_dictionary["batch_size"] = [1]
    hyperparamethers_range_dictionary["initialize"] = ["zero", "one", "random"]

    logFile = open(logFilePath + "Lambda_BPR_Cython" + "_BayesianSearch.txt", "a")

    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



    best_parameters = parameterSearch.search(recommenderDictionary, logFile = logFile, parallelize=False)

    logFile.write("best_parameters: {}".format(best_parameters))
    logFile.flush()
    logFile.close()

    pickle.dump(best_parameters, open(logFilePath + "Lambda_BPR_Cython" + "_best_parameters", "wb"), protocol=pickle.HIGHEST_PROTOCOL)















import os


if __name__ == '__main__':




    dataSplitter = DataSplitter_Warm(Movielens1MReader)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()





    runParameterSearch(URM_train, URM_validation)
