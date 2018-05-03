#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created by massimo
"""

import subprocess
import os, sys, time
import numpy as np
import scipy.sparse as sps
from numpy.linalg.linalg import LinAlgError

from Base.Recommender import Recommender
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from Base.Recommender_utils import check_matrix


class Lambda_BPR_Cython (Similarity_Matrix_Recommender, Recommender):

    RECOMMENDER_NAME = "Lambda_BPR_Cython"



    def __init__(self, URM_train, recompile_cython=False,
                 check_stability=False, save_lambda=False, save_eval=False):

        super(Lambda_BPR_Cython, self).__init__()

        self.save_lambda = save_lambda
        self.save_eval = save_eval
        self.URM_train = check_matrix(URM_train, "csr")
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.sparse_weights = True

        self.filterTopPop = False
        self.URM_mask = self.URM_train
        self.check_stability = check_stability

        #
        # if self.sparse_weights:
        #     self.S = sps.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        # else:
        #     self.S = np.zeros((self.n_users, self.n_users)).astype('float32')

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def launch_evaluation(self, URM_test):

        self.check_stability = False
        self.sparse_weights = True

        self.lambda_incremental = self.cythonEpoch.get_lambda()
        self.W_sparse_incremental = self.cythonEpoch.get_W_sparse(self.topK)
        self.W_sparse = self.W_sparse_incremental



        return self.evaluateRecommendations(URM_test)






    def fit_alreadyInitialized(self, epochs=30, URM_validation=None,
                               batch_size=1000, validation_every_n=1,
                               stop_on_validation = True, lower_validatons_allowed = 2, validation_metric = "map"):

        self.batch_size = batch_size
        start_time_train = time.time()

        #------------
        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0
        self.lambda_incremental = self.cythonEpoch.get_lambda()
        self.lambda_best = self.lambda_incremental.copy()
        self.W_sparse_incremental = self.cythonEpoch.get_W_sparse(self.topK)
        self.W_sparse_best = self.W_sparse_incremental.copy()


        #------------
        for currentEpoch in range(epochs+1):

            if convergence == True:
                break

            start_time_epoch = time.time()

            if currentEpoch > 0:
                if self.batch_size > 0:
                    self.cythonEpoch.epochIteration_Cython()
                else:
                    print("Error")
            else:
                #init in the 0 epoch
                self.cythonEpoch.epochIteration_Cython()

            if (URM_validation is not None) and (currentEpoch % validation_every_n == 0):

                print("Evaluation begins")

                results_run = self.launch_evaluation(URM_validation)
                self.doSaveLambdaAndEvaluate(currentEpoch, results_run)

                #-------- early stopping
                if stop_on_validation:
                    current_metric_value = results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value

                        self.lambda_best = self.lambda_incremental.copy()
                        self.W_sparse_best = self.W_sparse_incremental.copy()
                        self.epochs_best = currentEpoch
                        lower_validatons_count = 0

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print(
                            "SLIM_lambda_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                                currentEpoch + 1, validation_metric, self.epochs_best, best_validation_metric,
                                (time.time() - start_time_epoch) / 60))

                # If no validation required, always keep the latest
                else:
                    self.lambda_best = self.lambda_incremental.copy()
                    self.W_sparse_best = self.W_sparse_incremental.copy()
                    self.epochs_best = currentEpoch
                #------------


            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time() - start_time_epoch) / 60))

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))
        sys.stdout.flush()



    def fit(self, epochs=30, URM_validation=None, minRatingsPerUser=1, topK = 300,
            batch_size=1, validation_every_n=1,
            lambda_2=0, learning_rate=0.0002, sgd_mode='sgd', initialize = "zero", rcond=0.2, k=10,
            pseudoInv=False, lower_validatons_allowed=10, low_ram=True, force_positive = False):

        self.topK = topK
        self.rcond = rcond
        self.pseudoInv = pseudoInv
        self.sgd_mode = sgd_mode
        self.force_positive = force_positive
        #
        # if self.pseudoInv:
        #     #self.pinv = np.linalg.pinv(self.URM_train.todense(), rcond = rcond) # calculate pseudoinv if pseudoinv is enabled
        #     #print("singular values", np.linalg.svd(self.URM_train.todense(), compute_uv=False))
        #
        #     # U, s, Vh = scipy.sparse.linalg.svds(self.URM_train, k=10)
        #     # SVD_decomposition = (U, s, Vh)
        #

        self.eligibleUsers = []

        # Select only positive interactions
        URM_train = self.URM_train

        for user_id in range(self.n_users):
            start_pos = URM_train.indptr[user_id]
            end_pos = URM_train.indptr[user_id + 1]
            if len(URM_train.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)  #user that can be sampled

        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.initialize = initialize
        self.lambda_2 = lambda_2
        self.learning_rate = learning_rate

        from Lambda.Cython.Lambda_Cython import Lambda_BPR_Cython_Epoch

        # Cython
        if self.pseudoInv:

            try :
                self.cythonEpoch = Lambda_BPR_Cython_Epoch(self.URM_mask, self.URM_train, self.eligibleUsers, learning_rate=learning_rate, batch_size=batch_size, sgd_mode=sgd_mode,
                                                           lambda_2=lambda_2, enablePseudoInv=self.pseudoInv, low_ram = low_ram, initialize=initialize, rcond=rcond, k=k, force_positive = force_positive)

                self.fit_alreadyInitialized(epochs=epochs, URM_validation=URM_validation, batch_size=batch_size,
                                            validation_every_n=validation_every_n,
                                            lower_validatons_allowed=lower_validatons_allowed)

            except LinAlgError as linAlgError:

                print("SLIM_lambda_Cython: LinAlgError, SVD did not converge! Terminating...")

                raise linAlgError


        else:
            self.cythonEpoch = Lambda_BPR_Cython_Epoch(self.URM_mask, self.URM_train, self.eligibleUsers, learning_rate=learning_rate,
                                                       batch_size=batch_size, sgd_mode=sgd_mode, lambda_2=lambda_2, enablePseudoInv=self.pseudoInv, initialize=initialize, force_positive=force_positive)

            self.fit_alreadyInitialized(epochs=epochs, URM_validation=URM_validation, batch_size=batch_size,
                                        validation_every_n=validation_every_n, lower_validatons_allowed=lower_validatons_allowed)



    def runCompilationScript(self):

        compiledModuleSubfolder = "/Recommenders/Lambda/Cython"
        fileToCompile_list = ['Lambda_Cython.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python', 'compileCython.py', fileToCompile, 'build_ext', '--inplace' ]
            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)
            try:
                command = ['cython', fileToCompile, '-a']
                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)
            except:
                pass

        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        #python compileCython.py Lambda_Cython.pyx build_ext --inplace



    def doSaveLambdaAndEvaluate(self,currentEpoch, results_run):
        # Saving lambdas on file

        current_config = {'learn_rate': self.learning_rate, 'epoch': currentEpoch, 'sgd_mode': self.sgd_mode}
        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        if self.save_lambda:
            np.savetxt('out/lambdas/lambda_2' + str(self.lambda_2) + 'learning_rate' + str(self.learning_rate) + 'epoch' + str(
                currentEpoch) + '.txt', self.lambda_incremental, delimiter=',')

        #Saving evaluation on file
        if self.save_eval:
            t = "_transpose"
            if self.pseudoInv:
                t = "_pseudoinv_rcond"+str(self.rcond)

            with open('out/evaluations/'+"nnz_"+str(self.URM_train.nnz)+str(t)+'_lambda_2' + str(self.lambda_2) + '_learning_rate' + str(self.learning_rate) +"_"+ str(self.sgd_mode) +"_"+str(self.initialize)+ '.txt', 'a') as out:
                out.write('Epoch: ' + str(currentEpoch) +' ==> '+ str(results_run) + '\n')

    


    def get_lambda(self):

        return self.cythonEpoch.get_lambda()



    #
    #
    #
    #
    # def saveModel(self, folderPath, namePrefix = None):
    #
    #
    #     print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folderPath))
    #
    #     if namePrefix is None:
    #         namePrefix = self.RECOMMENDER_NAME
    #
    #         namePrefix += "_"
    #
    #     np.savez(folderPath + "{}.npz".format(namePrefix), user_lambda = self.get_lambda())
    #
    #
    #
    #
    # def loadModel(self, folderPath, namePrefix = None):
    #
    #     print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))
    #
    #     if namePrefix is None:
    #         namePrefix = self.RECOMMENDER_NAME
    #
    #         namePrefix += "_"
    #
    #
    #     npzfile = np.load(folderPath + "{}.npz".format(namePrefix))
    #
    #     for attrib_name in npzfile.files:
    #          self.__setattr__(attrib_name, npzfile[attrib_name])
    #





    def saveModel(self, folderPath, namePrefix = None, forceSparse = True):

        import pickle

        print("{}: Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME


        data_dict = {
            "sparse_weights":self.sparse_weights,
            "W_sparse":self.W_sparse,
            "user_lambda":self.get_lambda()
        }

        pickle.dump(data_dict,
                    open(folderPath + namePrefix, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)



    def loadModel(self, folderPath, namePrefix = None, forceSparse = True):

        import pickle

        print("{}: Loading model from folder '{}'".format(self.RECOMMENDER_NAME, folderPath))

        if namePrefix is None:
            namePrefix = self.RECOMMENDER_NAME


        data_dict = pickle.load(open(folderPath + namePrefix, "rb"))

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


