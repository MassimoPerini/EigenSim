#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import similarityMatrixTopK
from Base.Recommender import Recommender
import subprocess
import os, sys
import time
import numpy as np



def default_validation_function(self):


    return self.evaluateRecommendations(self.URM_validation)




class MF_BPR_Cython(Recommender):

    RECOMMENDER_NAME = "MF_BPR_Cython_Recommender"


    def __init__(self, URM_train, positive_threshold=4, URM_validation = None, recompile_cython = False):


        super(MF_BPR_Cython, self).__init__()


        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False

        self.positive_threshold = positive_threshold

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, filterCustomItems = np.array([], dtype=np.int), minRatingsPerUser=1,
            batch_size = 1000, num_factors=10,
            learning_rate = 0.01, sgd_mode='sgd', user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "map",
            validation_function = None, validation_every_n = 1):



        self.num_factors = num_factors


        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()

        self.sgd_mode = sgd_mode


        # Import compiled module
        from MatrixFactorization.Cython.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch


        self.cythonEpoch = MF_BPR_Cython_Epoch(URM_train_positive,
                                                 n_factors = self.num_factors,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode = sgd_mode,
                                                 user_reg=user_reg,
                                                 positive_reg=positive_reg,
                                                 negative_reg=negative_reg)


        if validation_function is None:
            validation_function = default_validation_function


        self.batch_size = batch_size
        self.learning_rate = learning_rate


        start_time = time.time()



        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.W_incremental = self.cythonEpoch.get_W()
        self.W_best = self.W_incremental.copy()

        self.H_incremental = self.cythonEpoch.get_H()
        self.H_best = self.H_incremental.copy()

        self.epochs_best = 0

        currentEpoch = 0

        while currentEpoch < epochs and not convergence:

            if self.batch_size>0:
                self.cythonEpoch.epochIteration_Cython()
            else:
                print("No batch not available")

            # Determine whether a validaton step is required
            if self.URM_validation is not None and (currentEpoch + 1) % validation_every_n == 0:

                print("MF_BPR_Cython: Validation begins...")

                self.W_incremental = self.cythonEpoch.get_W()
                self.H_incremental = self.cythonEpoch.get_H()

                self.W = self.W_incremental.copy()
                self.H = self.H_incremental.copy()

                results_run = validation_function(self)

                print("MF_BPR_Cython: {}".format(results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value

                        self.W_best = self.W_incremental.copy()
                        self.H_best = self.H_incremental.copy()

                        self.epochs_best = currentEpoch +1
                        lower_validatons_count = 0

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print("MF_BPR_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                            currentEpoch+1, validation_metric, self.epochs_best, best_validation_metric, (time.time() - start_time) / 60))


            # If no validation required, always keep the latest
            if not stop_on_validation:
                self.W_best = self.W_incremental.copy()
                self.H_best = self.H_incremental.copy()

            print("MF_BPR_Cython: Epoch {} of {}. Elapsed time {:.2f} min".format(
                currentEpoch+1, epochs, (time.time() - start_time) / 60))

            currentEpoch += 1


        self.W = self.W_best.copy()
        self.H = self.H_best.copy()


        sys.stdout.flush()





    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/MatrixFactorization/Cython"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        #python compileCython.py MF_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "MF_BPR_Cython_Epoch.pyx"])






    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()




    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]

        scores_array = np.dot(self.W[users_in_batch], self.H.T)

        if self.normalize:
            raise ValueError("Not implemented")

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
            scores_array[user_profile_batch.nonzero()] = -np.inf

        if filterTopPop:
            scores_array[:,self.filterTopPop_ItemsID] = -np.inf

        if filterCustomItems:
            scores_array[:, self.filterCustomItems_ItemsID] = -np.inf


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = (-scores_array).argsort(axis=1)
        #ranking = np.fliplr(ranking)
        #ranking = ranking[:,0:n]

        ranking = np.zeros((scores_array.shape[0],n), dtype=np.int)

        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]

            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]


        return ranking



    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):


        if n==None:
            n=self.URM_train.shape[1]-1

        scores_array = np.dot(self.W[user_id], self.H.T)

        if self.normalize:
            raise ValueError("Not implemented")


        if exclude_seen:
            scores = self._filter_seen_on_scores(user_id, scores_array)

        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores_array)

        if filterCustomItems:
            scores = self._filterCustomItems_on_scores(scores_array)


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]


        return ranking






    def saveModel(self, folderPath, namePrefix = None, forceSparse = True):

        print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        np.savez(folderPath + "{}.npz".format(namePrefix), W = self.W, H = self.H)



    def loadModel(self, folderPath, namePrefix = None, forceSparse = True):


        print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME

        namePrefix += "_"

        npzfile = np.load(folderPath + "{}.npz".format(namePrefix))

        for attrib_name in npzfile.files:
             self.__setattr__(attrib_name, npzfile[attrib_name])
