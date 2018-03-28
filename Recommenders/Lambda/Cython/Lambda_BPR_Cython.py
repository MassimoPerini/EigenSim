#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created by massimo
"""

import subprocess
import os, sys, time
import numpy as np
import scipy.sparse as sps
from scipy import linalg
from Recommenders.utils import *


class Lambda_BPR_Cython:
    def __init__(self, URM_train, recompile_cython=False, sgd_mode='sgd', pseudoInv=False, rcond = 0.2, check_stability = False, save_lambda = False, save_eval = True):
        self.save_lambda = save_lambda
        self.save_eval = save_eval
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.sparse_weights = True
        self.rcond = rcond
        self.filterTopPop = False
        if pseudoInv:
            self.pinv = np.linalg.pinv(self.URM_train.todense(), rcond = rcond) # calculate pseudoinv if pseudoinv is enabled
            #print("singular values", np.linalg.svd(self.URM_train.todense(), compute_uv=False))
        self.pseudoInv = pseudoInv
        #self.URM_mask = self.URM_train.copy()
        self.URM_mask = self.URM_train
        if self.sparse_weights:
            self.S = sps.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        else:
            self.S = np.zeros((self.n_users, self.n_users)).astype('float32')
        self.sgd_mode = sgd_mode
        self.check_stability = check_stability
        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    #------------EVALUATION
    def evaluateRecommendations(self, URM_test, at=5, minRatingsPerUser=1, exclude_seen=True, pseudoInverse = False):
        check_stability = self.check_stability
        self.URM_test = check_matrix(URM_test, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        if pseudoInverse == False:
            self.similarity = (self.URM_train.transpose().dot(self.W_sparse.dot(self.URM_train)))
            self.similarity.eliminate_zeros()
            self.similarity = check_matrix(self.similarity, format='csr')
        else:
            self.similarity = self.pinv.dot((self.W_sparse.dot(self.URM_train)).todense())
            #self.similarity = check_matrix(self.similarity, format='csc')
            print("similarity matrix: ", self.similarity.shape)

        nusers = self.URM_test.shape[0]
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]
        usersToEvaluate = list(usersToEvaluate)
        if pseudoInverse == False:
            print("users to test: ", len(usersToEvaluate), "non-zero similarity-sparse: ", self.similarity.nnz)
        return self.evaluateRecommendationsSequential(usersToEvaluate, check_stability=check_stability, pseudoInv = pseudoInverse)

    def get_user_relevant_items(self, user_id):
        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def get_user_test_ratings(self, user_id):
        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id + 1]]

    def evaluateRecommendationsSequential(self, usersToEvaluate, check_stability = True, pseudoInv = False):
        start_time = time.time()
        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        roc_auc_tmp, precision_tmp, recall_tmp, map_tmp, mrr_tmp, ndcg_tmp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        roc_auc_delta, precision_delta, recall_delta, map_delta, mrr_delta, ndcg_delta = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        n_eval = 0

        for test_user in usersToEvaluate: #for each valid user
            if n_eval % 1000 == 0:
                print(n_eval)
            relevant_items = self.get_user_relevant_items(test_user)
            n_eval += 1
            recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen, n=self.at, test_stability_ranking=False, pseudoInv=pseudoInv)
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True) #how many are good
            roc_auc_tmp = roc_auc(is_relevant)
            precision_tmp = precision(is_relevant)
            recall_tmp = recall(is_relevant, relevant_items)
            map_tmp = map(is_relevant, relevant_items)
            mrr_tmp = rr(is_relevant)
            ndcg_tmp = ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)

            roc_auc_ += roc_auc_tmp
            precision_ += precision_tmp
            recall_ += recall_tmp
            map_ += map_tmp
            mrr_ += mrr_tmp
            ndcg_ += ndcg_tmp

            if check_stability: #if I want to check the stability
                relevant_items = self.get_user_relevant_items(test_user)
                #esegui la predizione con il profilo "modificato" e calcola le differenze
                recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen, n=self.at, test_stability_ranking=True, pseudoInv=pseudoInv)
                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
                roc_auc_delta += (roc_auc(is_relevant) - roc_auc_tmp)
                precision_delta += (precision(is_relevant) - precision_tmp)
                recall_delta += (recall(is_relevant, relevant_items) - recall_tmp)
                map_delta += (map(is_relevant, relevant_items) - map_tmp)
                mrr_delta += (rr(is_relevant) -  mrr_tmp)
                ndcg_delta += (ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at) - ndcg_tmp)

            if (n_eval % 10000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                    n_eval,
                    100.0 * float(n_eval) / len(usersToEvaluate),
                    time.time() - start_time,
                    float(n_eval) / (time.time() - start_time)))
        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

            if check_stability:
                roc_auc_delta /= n_eval
                precision_delta /= n_eval
                recall_delta /= n_eval
                map_delta /= n_eval
                mrr_delta /= n_eval
                ndcg_delta /= n_eval


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}
        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        if check_stability:
            results_run["AUC_delta"] = roc_auc_delta
            results_run["precision_delta"] = precision_delta
            results_run["recall_delta"] = recall_delta
            results_run["map_delta"] = map_delta
            results_run["NDCG_delta"] = ndcg_delta
            results_run["MRR_delta"] = mrr_delta

        return (results_run)

    #----------------RECOMMEND

    def recommend(self, user_id, n=None, exclude_seen=True, test_stability_ranking = False, pseudoInv = False):
        if self.sparse_weights:
            if pseudoInv:
                #user_profile = self.URM_train[user_id].todense()
                user_profile = self.URM_train[user_id]
                scores = np.ravel(user_profile.dot(self.similarity))
            else:
                user_profile = self.URM_train[user_id]
                scores = user_profile.dot(self.similarity).toarray().ravel()
        else:
            # necessita di modifiche (matrici dense non sono state usate)
            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        if exclude_seen: #exclude items already seen
            scores = self._filter_seen_on_scores(user_id, scores)

        #partiziona in base all'n-esimo elemento (in ordine decrescente)
        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting] #genera il ranking

        #se devo testare la stabilità devo eseguire il calcolo con il profilo modificato
        if test_stability_ranking:
            ranking = np.asarray(ranking)
            ranking = ranking[:2] # prendi i primi 2 elementi ed inseriscili nel profilo (elementi ad 1)
            user_profile = self.URM_train[user_id].copy()
            user_profile[0, ranking] = 1
            if pseudoInv:
                scores = np.ravel(user_profile.dot(self.similarity))
            else:
                scores = user_profile.dot(self.similarity).toarray().ravel() # ri-calcola i punteggi

            if exclude_seen: # evita di predire gli elementi visti
                seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
                scores[seen] = -np.inf
                scores[ranking] = -np.inf
            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

    def _filter_seen_on_scores(self, user_id, scores):
        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    #---------FIT

    def epochIteration(self):
        self.S = self.cythonEpoch.epochIteration_Cython()
        if self.sparse_weights:
            self.W_sparse = self.S
        else:
            self.W = self.S

    #do the iterations
    def fit_alreadyInitialized(self, epochs=30, logFile=None, URM_test=None, minRatingsPerUser=1,
                                batch_size=1000, validate_every_N_epochs=1, start_validation_after_N_epochs=0,
                                alpha=0.00025, learning_rate=0.0005):

        self.batch_size = batch_size
        self.alpha = alpha
        start_time_train = time.time()

        for currentEpoch in range(epochs):
            start_time_epoch = time.time()
            if currentEpoch > 0:
                if self.batch_size > 0:
                    self.epochIteration()
                else:
                    print("Error")
            else:
                #init in the 0 epoch
                if self.sparse_weights:
                    self.W_sparse = self.S
                else:
                    self.W = self.S

                #results_run = self.evaluateRecommendations(URM_test, minRatingsPerUser=minRatingsPerUser)
                #self.doSaveLambdaAndEvaluate(-1, results_run)
                self.epochIteration()
            if (URM_test is not None) and (currentEpoch % validate_every_N_epochs == 0 or (currentEpoch == 0)) and currentEpoch >= start_validation_after_N_epochs:
                print("Evaluation begins")
                results_run = self.evaluateRecommendations(URM_test, minRatingsPerUser=minRatingsPerUser, pseudoInverse=self.pseudoInv)

                self.doSaveLambdaAndEvaluate(currentEpoch, results_run)
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time() - start_time_epoch) / 60))
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch+1, epochs, float(time.time() - start_time_epoch) / 60))
        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))
        sys.stdout.flush()


    def fit(self, epochs=30, logFile=None, URM_test=None, minRatingsPerUser=1,
            batch_size=1, validate_every_N_epochs=1, start_validation_after_N_epochs=0,
            alpha=0.00025, learning_rate=0.0002, sgd_mode='sgd', initialize = "zero"):
        self.eligibleUsers = []
        # Select only positive interactions
        URM_train = self.URM_train
        for user_id in range(self.n_users):
            start_pos = URM_train.indptr[user_id]
            end_pos = URM_train.indptr[user_id + 1]
            if len(URM_train.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)  #user that can be sampled
        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode
        self.initialize = initialize
        self.alpha = alpha
        self.learning_rate = learning_rate

        # Cython
        from Recommenders.Lambda.Cython.Lambda_Cython import Lambda_BPR_Cython_Epoch
        if self.pseudoInv:
            self.cythonEpoch = Lambda_BPR_Cython_Epoch(self.URM_mask, self.sparse_weights, self.eligibleUsers, learning_rate=learning_rate, batch_size=batch_size, sgd_mode=sgd_mode, alpha=alpha, enablePseudoInv=self.pseudoInv, pseudoInv=self.pinv, initialize=initialize)
            self.fit_alreadyInitialized(epochs=epochs, logFile=logFile, URM_test=URM_test, minRatingsPerUser=minRatingsPerUser, batch_size=batch_size,
                                                            validate_every_N_epochs=validate_every_N_epochs, start_validation_after_N_epochs=start_validation_after_N_epochs,
                                                            learning_rate=learning_rate, alpha=alpha)

        else:
            self.cythonEpoch = Lambda_BPR_Cython_Epoch(self.URM_mask, self.sparse_weights, self.eligibleUsers, learning_rate=learning_rate, batch_size=batch_size, sgd_mode=sgd_mode, alpha=alpha, enablePseudoInv=self.pseudoInv, initialize=initialize)
            self.fit_alreadyInitialized(epochs=epochs, logFile=logFile, URM_test=URM_test, minRatingsPerUser=minRatingsPerUser, batch_size=batch_size,
                                                            validate_every_N_epochs=validate_every_N_epochs, start_validation_after_N_epochs=start_validation_after_N_epochs,
                                                            learning_rate=learning_rate, alpha=alpha)

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

    def doSaveLambdaAndEvaluate(self,currentEpoch, results_run):
        # Saving lambdas on file

        current_config = {'learn_rate': self.learning_rate, 'epoch': currentEpoch, 'sgd_mode': self.sgd_mode}
        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        if self.save_lambda:
            np.savetxt('out/lambdas/alpha' + str(self.alpha) + 'learning_rate' + str(self.learning_rate) + 'epoch' + str(
                currentEpoch) + '.txt', self.W_sparse.diagonal(), delimiter=',')

        #Saving evaluation on file
        if self.save_eval:
            t = "_transpose"
            if self.pseudoInv:
                t = "_pseudoinv_rcond"+str(self.rcond)

            with open('out/evaluations/'+"nnz_"+str(self.URM_train.nnz)+str(t)+'_alpha' + str(self.alpha) + '_learning_rate' + str(self.learning_rate) +"_"+ str(self.sgd_mode) +"_"+str(self.initialize)+ '.txt', 'a') as out:
                out.write('Epoch: ' + str(currentEpoch) +' ==> '+ str(results_run) + '\n')

    

