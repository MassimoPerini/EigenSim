#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import multiprocessing
import time

import numpy as np

from Base.metrics import roc_auc, precision, recall, map, ndcg, rr, arhr
#from Base.Cython.metrics import roc_auc, precision, recall, map, ndcg, rr
from Base.Recommender_utils import check_matrix, areURMequals, removeTopPop


class Recommender(object):
    """Abstract Recommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self):
        super(Recommender, self).__init__()
        self.URM_train = None
        self.sparse_weights = True
        self.normalize = False

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.filterCustomItems = False
        self.filterCustomItems_ItemsID = np.array([], dtype=np.int)


    def fit(self):
        pass

    def _filter_TopPop_on_scores(self, scores):
        scores[self.filterTopPop_ItemsID] = -np.inf
        return scores


    def _filterCustomItems_on_scores(self, scores):
        scores[self.filterCustomItems_ItemsID] = -np.inf
        return scores


    def _filter_seen_on_scores(self, user_id, scores):

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores



    def saveModel(self, folderPath, namePrefix = None):
        raise NotImplementedError("Recommender: saveModel not implemented")


    def loadModel(self, folderPath, namePrefix = None):
        raise NotImplementedError("Recommender: loadModel not implemented")


    def evaluateRecommendations(self, URM_test_new, at=5, minRatingsPerUser=1, exclude_seen=True,
                                mode='sequential', filterTopPop = False,
                                filterCustomItems = np.array([], dtype=np.int),
                                filterCustomUsers = np.array([], dtype=np.int)):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test_new:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :param filterCustomItems: Array, default empty           Items ID to NOT take into account when recommending
        :param filterCustomUsers: Array, default empty           Users ID to NOT take into account when recommending
        :return:
        """

        if len(filterCustomItems) == 0:
            self.filterCustomItems = False
        else:
            self.filterCustomItems = True
            self.filterCustomItems_ItemsID = np.array(filterCustomItems)


        if filterTopPop != False:

            self.filterTopPop = True

            _,_, self.filterTopPop_ItemsID = removeTopPop(self.URM_train, URM_2 = URM_test_new, percentageToRemove=filterTopPop)

            print("Filtering {}% TopPop items, count is: {}".format(filterTopPop*100, len(self.filterTopPop_ItemsID)))

            # Zero-out the items in order to be considered irrelevant
            URM_test_new = check_matrix(URM_test_new, format='lil')
            URM_test_new[:,self.filterTopPop_ItemsID] = 0
            URM_test_new = check_matrix(URM_test_new, format='csr')


        # During testing CSR is faster
        self.URM_test = check_matrix(URM_test_new, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen


        nusers = self.URM_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]

        if len(filterCustomUsers) != 0:
            print("Filtering {} Users".format(len(filterCustomUsers)))
            usersToEvaluate = set(usersToEvaluate) - set(filterCustomUsers)

        usersToEvaluate = list(usersToEvaluate)



        if mode=='sequential':
            return self.evaluateRecommendationsSequential(usersToEvaluate)
        elif mode=='parallel':
            return self.evaluateRecommendationsParallel(usersToEvaluate)
        elif mode=='batch':
            return self.evaluateRecommendationsBatch(usersToEvaluate)
        elif mode=='cython':
             return self.evaluateRecommendationsCython(usersToEvaluate)
        # elif mode=='random-equivalent':
        #     return self.evaluateRecommendationsRandomEquivalent(usersToEvaluate)
        else:
            raise ValueError("Mode '{}' not available".format(mode))


    def get_user_relevant_items(self, user_id):

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]

    def get_user_test_ratings(self, user_id):

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def evaluateRecommendationsCython(self, usersToEvaluate):

        # Command to run compilation script
        #python compileCython.py build_ext --inplace

        from Base.Cython.Similarity_Matrix_Evaluator import Similarity_Matrix_Evaluator

        if self.sparse_weights:
            SimilarityMatrix = self.W_sparse#.toarray()
        else:
            SimilarityMatrix = self.W

        evaluator = Similarity_Matrix_Evaluator(SimilarityMatrix, self.URM_test,
                                                self.URM_train,
                                                filterTopPop = self.filterTopPop, filterTopPop_ItemsID=np.array(self.filterTopPop_ItemsID, dtype=np.int32),
                                                normalize=self.normalize)

        return evaluator.evaluateRecommendations(np.array(usersToEvaluate, dtype=np.int32), at=self.at, exclude_seen=self.exclude_seen)

    #
    #
    # def evaluateRecommendationsSequential_multipleURM(self, usersToEvaluate, URM_list):
    #
    #     results_run_list = []
    #
    #     for _ in range(len(URM_list)):
    #
    #         results_run = {"AUC": 0.0,
    #                        "precision": 0.0,
    #                        "recall": 0.0,
    #                        "map": 0.0,
    #                        "MRR": 0.0,
    #                        "NDCG": 0.0,
    #                        "n_eval": 0
    #                        }
    #
    #         results_run_list.append(results_run)
    #
    #     test_user_count = 0
    #     start_time = time.time()
    #     start_time_print = time.time()
    #
    #     for test_user in usersToEvaluate:
    #
    #         test_user_count += 1
    #
    #         # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower
    #
    #         recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
    #                                            n=self.at, filterTopPop=self.filterTopPop, filterCustomItems=self.filterCustomItems)
    #
    #         for urm_index in range(len(URM_list)):
    #
    #             currentURM = URM_list[urm_index]
    #
    #             relevant_items = currentURM.indices[currentURM.indptr[test_user]:currentURM.indptr[test_user+1]]
    #
    #             is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    #
    #             # evaluate the recommendation list with ranking metrics ONLY
    #             results_run_list[urm_index]["AUC"] += roc_auc(is_relevant)
    #             results_run_list[urm_index]["precision"] += precision(is_relevant)
    #             results_run_list[urm_index]["recall"] += recall(is_relevant, relevant_items)
    #             results_run_list[urm_index]["map"] += map(is_relevant, relevant_items)
    #             results_run_list[urm_index]["MRR"] += rr(is_relevant)
    #             results_run_list[urm_index]["NDCG"] += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)
    #
    #             results_run_list[urm_index]["n_eval"] += 1
    #
    #
    #         if time.time() - start_time_print >= 30 or test_user_count==len(usersToEvaluate)-1:
    #             print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
    #                               test_user_count,
    #                               100.0* float(test_user_count+1)/len(usersToEvaluate),
    #                               time.time()-start_time,
    #                               float(test_user_count)/(time.time()-start_time)))
    #
    #             start_time_print = time.time()
    #
    #
    #     for urm_index in range(len(URM_list)):
    #
    #         n_eval = results_run_list[urm_index]["n_eval"]
    #
    #         if (n_eval > 0):
    #             results_run_list[urm_index]["AUC"] /= n_eval
    #             results_run_list[urm_index]["precision"] /= n_eval
    #             results_run_list[urm_index]["recall"] /= n_eval
    #             results_run_list[urm_index]["map"] /= n_eval
    #             results_run_list[urm_index]["MRR"] /= n_eval
    #             results_run_list[urm_index]["NDCG"] /= n_eval
    #
    #         else:
    #             print("WARNING: No users had a sufficient number of relevant items")
    #
    #
    #     return (results_run_list)


    def evaluateRecommendationsSequential(self, usersToEvaluate):

        start_time = time.time()
        start_time_print = time.time()

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_, f1_, hit_rate_, arhr_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        for test_user in usersToEvaluate:

            # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)

            n_eval += 1

            recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                               n=self.at, filterTopPop=self.filterTopPop, filterCustomItems=self.filterCustomItems)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            # evaluate the recommendation list with ranking metrics ONLY
            hit_rate_ += is_relevant.sum()
            roc_auc_ += roc_auc(is_relevant)
            precision_ += precision(is_relevant)
            recall_ += recall(is_relevant, relevant_items)
            map_ += map(is_relevant, relevant_items)
            mrr_ += rr(is_relevant)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)
            arhr_ += arhr(is_relevant)




            if time.time() - start_time_print > 30 or n_eval==len(usersToEvaluate)-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval+1)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))

                start_time_print = time.time()


        if (n_eval > 0):
            hit_rate_ /= n_eval
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval
            arhr_ /= n_eval
            f1_ = 2 * (precision_ * recall_) / (precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_
        results_run["F1"] = f1_
        results_run["HR"] = hit_rate_
        results_run["ARHR"] = arhr_

        return (results_run)

    #
    # def evaluateRecommendationsRandomEquivalent_oneUser(self, test_user):
    #
    #     hitCount = 0
    #
    #     seenItems = set(self.URM_test_relevantItems[test_user])
    #     seenItems.union(set(self.URM_train_relevantItems[test_user]))
    #
    #     unseenItems = self.allItemsSet.difference(seenItems)
    #
    #     # Being the URM CSR, the indices are the non-zero column indexes
    #     user_profile = self.URM_train_user_profile[test_user]
    #
    #     # hits_vector = np.zeros(numRandomItems)
    #
    #
    #
    #     if self.sparse_weights:
    #         scores = user_profile.dot(self.W_sparse).toarray().ravel()
    #         # scores = self.scoresAll[user_id].toarray().ravel()
    #     else:
    #         scores = user_profile.dot(self.W).ravel()
    #
    #     ranking = scores.argsort()
    #     ranking = np.flip(ranking, axis=0)
    #     ranking = ranking[0:100]
    #
    #     # For each item
    #     for test_item in self.URM_test_relevantItems[test_user]:
    #
    #         # Randomly select a given number of items, default 1000
    #         other_items = random.sample(unseenItems, self.numRandomItems)
    #         other_items.append(test_item)
    #
    #         items_mask = np.in1d(ranking, other_items, assume_unique=True)
    #         ranking = ranking[items_mask]
    #
    #         item_position = np.where(ranking == test_item)
    #
    #         if len(item_position) > 0:
    #             # hits_vector[item_position:numRandomItems] += 1
    #             hitCount += 1
    #
    #     #print(test_user)
    #     self.evaluateRecommendationsRandomEquivalent_hit += hitCount
    #
    #
    # def evaluateRecommendationsRandomEquivalent(self, usersToEvaluate, numRandomItems = 1000):
    #
    #     start_time = time.time()
    #
    #     # Initialize data structure for unseen items
    #     nitems = self.URM_test.shape[1]
    #
    #
    #     self.allItemsSet = set(np.arange(nitems))
    #     self.numRandomItems = numRandomItems
    #     self.evaluateRecommendationsRandomEquivalent_hit = 0
    #
    #     print("Parallel evaluation starting")
    #
    #     #pool = multiprocessing.Pool(processes=2, maxtasksperchild=10)
    #     #pool.map(self.evaluateRecommendationsRandomEquivalent_oneUser, usersToEvaluate)
    #
    #     n_eval = 0
    #     for test_user in usersToEvaluate:
    #         self.evaluateRecommendationsRandomEquivalent_oneUser(test_user)
    #
    #         n_eval += 1
    #
    #         if(n_eval % 1000 == 0):
    #             print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
    #                               n_eval,
    #                               100.0* float(n_eval)/len(usersToEvaluate),
    #                               time.time()-start_time,
    #                               float(n_eval)/(time.time()-start_time)))
    #
    #
    #
    #
    #
    #     hitCount = self.evaluateRecommendationsRandomEquivalent_hit
    #
    #     print("Evaluation complete in {:.2f} seconds".format(time.time()-start_time))
    #
    #     recall_value = hitCount / self.URM_test.nnz
    #
    #     results_run = {}
    #
    #     results_run["AUC"] = 0.0
    #     results_run["precision"] = 0.0
    #     results_run["recall"] = recall_value
    #     results_run["map"] = 0.0
    #     results_run["NDCG"] = 0.0
    #     results_run["MRR"] = 0.0
    #
    #     return (results_run)
    #



    def evaluateRecommendationsBatch(self, usersToEvaluate, batch_size = 1000):

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        start_time = time.time()
        start_time_batch = time.time()

        #Number of blocks is rounded to the next integer
        totalNumberOfBatch = int(len(usersToEvaluate) / batch_size) + 1

        for current_batch in range(totalNumberOfBatch):

            user_first_id = current_batch*batch_size
            user_last_id = min((current_batch+1)*batch_size-1,  len(usersToEvaluate)-1)

            users_in_batch = usersToEvaluate[user_first_id:user_last_id]

            relevant_items_batch = self.URM_test[users_in_batch]

            recommended_items_batch = self.recommendBatch(users_in_batch,
                                                          exclude_seen=self.exclude_seen,
                                                          n=self.at, filterTopPop=self.filterTopPop,
                                                          filterCustomItems=self.filterCustomItems)


            for test_user in range(recommended_items_batch.shape[0]):

                n_eval += 1

                current_user = relevant_items_batch[test_user,:]

                relevant_items = current_user.indices
                recommended_items = recommended_items_batch[test_user,:]

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                # evaluate the recommendation list with ranking metrics ONLY
                roc_auc_ += roc_auc(is_relevant)
                precision_ += precision(is_relevant)
                recall_ += recall(is_relevant, relevant_items)
                map_ += map(is_relevant, relevant_items)
                mrr_ += rr(is_relevant)
                ndcg_ += ndcg(recommended_items, relevant_items, relevance=current_user.data, at=self.at)



            if(time.time() - start_time_batch >= 30 or current_batch == totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))

                start_time_batch = time.time()


        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)



    def evaluateOneUser(self, test_user):

        # Being the URM CSR, the indices are the non-zero column indexes
        #relevant_items = self.URM_test_relevantItems[test_user]
        relevant_items = self.URM_test[test_user].indices

        # this will rank top n items
        recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                           n=self.at, filterTopPop=self.filterTopPop,
                                           filterCustomItems=self.filterCustomItems)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        hit_rate_ = is_relevant.sum()
        roc_auc_ = roc_auc(is_relevant)
        precision_ = precision(is_relevant)
        recall_ = recall(is_relevant, relevant_items)
        map_ = map(is_relevant, relevant_items)
        mrr_ = rr(is_relevant)
        ndcg_ = ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)
        arhr_ = arhr(is_relevant)

        return roc_auc_, precision_, recall_, map_, mrr_, ndcg_, hit_rate_, arhr_



    def evaluateRecommendationsParallel(self, usersToEvaluate):

        print("Evaluation of {} users begins".format(len(usersToEvaluate)))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        resultList = pool.map(self.evaluateOneUser, usersToEvaluate)

        #for i, _ in enumerate(pool.imap_unordered(self.evaluateOneUser, usersToEvaluate), 1):
        #    if(i%1000 == 0):
        #        sys.stderr.write('\rEvaluated {} users ({0:%})'.format(i , i / usersToEvaluate))

        # Close the pool to avoid memory leaks
        pool.close()

        n_eval = len(usersToEvaluate)
        roc_auc_, precision_, recall_, map_, mrr_, ndcg_, f1_, hit_rate_, arhr_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


        # Looping is slightly faster then using the numpy vectorized approach, less data transformation
        for result in resultList:
            roc_auc_ += result[0]
            precision_ += result[1]
            recall_ += result[2]
            map_ += result[3]
            mrr_ += result[4]
            ndcg_ += result[5]
            hit_rate_ += result[6]
            arhr_ += result[7]


        if (n_eval > 0):
            hit_rate_ /= n_eval
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval
            arhr_ /= n_eval
            f1_ = 2 * (precision_ * recall_) / (precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_
        results_run["F1"] = f1_
        results_run["HR"] = hit_rate_
        results_run["ARHR"] = arhr_

        return (results_run)



