#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import zipfile

from data.DataReader import DataReader, removeFeatures, reconcile_mapper_with_removed_tokens
from Recommenders.Base.Recommender_utils import check_matrix
from data.URM_Dense_K_Cores import select_k_cores


class NetflixEnhancedReader(DataReader):

    DATASET_SUBFOLDER = "NetflixEnhanced/"
    AVAILABLE_ICM = ["ICM_all", "ICM_tags", "ICM_editorial"]

    EDITORIAL_ICM = "ICM_editorial"


    def __init__(self, apply_k_cores = None):
        """
        :param splitSubfolder:
        """

        super(NetflixEnhancedReader, self).__init__(apply_k_cores = apply_k_cores)



    def load_from_original_file(self):

        # Load data from original

        print("NetflixEnhancedReader: Loading original data")

        zipFile_path =  "./data/" + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "NetflixEnhancedData.zip")

        except FileNotFoundError as fileNotFound:
            raise fileNotFound


        URM_matfile_path = dataFile.extract("urm.mat", path=zipFile_path + "NetflixEnhancedData/")
        URM_matfile = sio.loadmat(URM_matfile_path)

        self.URM_all = URM_matfile["urm"]
        usercache_urm = URM_matfile["usercache_urm"]
        itemcache_urm = URM_matfile["itemcache_urm"]

        self.set_global_mappers(usercache_urm, itemcache_urm)

        titles_matfile_path = dataFile.extract("titles.mat", path=zipFile_path + "NetflixEnhancedData/")
        titles_matfile = sio.loadmat(titles_matfile_path)

        titles_list = titles_matfile["titles"]

        URM_matfile_path = dataFile.extract("icm.mat", path=zipFile_path + "NetflixEnhancedData/")
        ICM_matfile = sio.loadmat(URM_matfile_path)

        self.ICM_all = ICM_matfile["icm"]
        self.ICM_all = check_matrix(self.ICM_all.T, 'csr')

        ICM_dictionary = ICM_matfile["dictionary"]
        itemcache_icm = ICM_matfile["itemcache_icm"]
        stemTypes = ICM_dictionary["stemTypes"][0][0]
        stems = ICM_dictionary["stems"][0][0]

        # Split ICM_tags and ICM_editorial
        is_tag_mask = np.zeros((len(stems)), dtype=np.bool)

        for current_stem_index in range(len(stems)):
            current_stem_type = stemTypes[current_stem_index]
            current_stem_type_string = current_stem_type[0][0]

            if "KeywordsArray" in current_stem_type_string:
                is_tag_mask[current_stem_index] = True


        self.ICM_tags = self.ICM_all[:,is_tag_mask]

        is_editorial_mask = np.logical_not(is_tag_mask)
        self.ICM_editorial = self.ICM_all[:, is_editorial_mask]


        # Eliminate items and users with no interactions or less than the desired value
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        n_items = self.ICM_all.shape[0]
        ICM_filter_mask = np.ones(n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False

        # Remove items in ICM as well to ensure consistency
        self.ICM_all = self.ICM_all[ICM_filter_mask,:]
        self.ICM_tags = self.ICM_tags[ICM_filter_mask,:]
        self.ICM_editorial = self.ICM_editorial[ICM_filter_mask,:]


        # Remove features taking into account the filtered ICM
        self.ICM_all, _ = removeFeatures(self.ICM_all, minOccurrence = 5, maxPercOccurrence = 0.30)
        self.ICM_tags, _ = removeFeatures(self.ICM_tags, minOccurrence = 5, maxPercOccurrence = 0.30)
        self.ICM_editorial, _ = removeFeatures(self.ICM_editorial, minOccurrence = 5, maxPercOccurrence = 0.30)


        print("NetflixEnhancedReader: saving URM_train and ICM")
        sps.save_npz(self.data_path + "URM_all{}.npz".format(self.k_cores_name_suffix), self.URM_all)

        for icm_to_save_name in self.AVAILABLE_ICM:
            sps.save_npz(self.data_path + "{}{}.npz".format(icm_to_save_name, self.k_cores_name_suffix), self.__getattribute__(icm_to_save_name))


        self.save_mappers()


        print("NetflixEnhancedReader: loading complete")



    def set_global_mappers(self, usercache_urm, itemcache_urm):

        for usercache_entry in usercache_urm:
            self.item_original_ID_to_index[usercache_entry[0]] = usercache_entry[1]


        for itemcache_entry in itemcache_urm:
            self.user_original_ID_to_index[itemcache_entry[0]] = itemcache_entry[1]





    def get_statistics(self):

        super(NetflixEnhancedReader, self).get_statistics()

        # n_genres = self.ICM_genres.shape[1]
        # n_tags = self.ICM_tags_train.shape[1]
        n_features = self.get_ICM().shape[1]
        n_items = self.get_ICM().shape[0]

        print("\tNumber of unique features (genres + tags): {}\n"
              "\tICM density: {:.2E}\n".format(
              n_features, self.get_ICM().nnz/(n_items*n_features)))




    def get_hyperparameters_for_rec_class(self, target_recommender):

        from KNN.item_knn_CBF import ItemKNNCBFRecommender
        from KNN.item_knn_CF import ItemKNNCFRecommender
        from KNN.user_knn_CF import UserKNNCFRecommender
        from SLIM_ElasticNet.SLIM_ElasticNet import MultiThreadSLIM_ElasticNet
        from SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet
        from GraphBased.P3alpha import P3alphaRecommender
        from GraphBased.RP3beta import RP3betaRecommender

        try:
            from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
            from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
            from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
        except ImportError:
            MF_BPR_Cython = None
            FunkSVD = None
            SLIM_BPR_Cython = None


        hyperparam_dict = {}

        if target_recommender is ItemKNNCBFRecommender:
            hyperparam_dict["topK"] = 50
            hyperparam_dict["shrink"] = 100
            hyperparam_dict["similarity"] = 'jaccard'
            hyperparam_dict["normalize"] = True

            return hyperparam_dict

        elif target_recommender is ItemKNNCFRecommender:
            hyperparam_dict["topK"] = 150
            hyperparam_dict["shrink"] = 0
            hyperparam_dict["similarity"] = 'cosine'
            hyperparam_dict["normalize"] = True

            return hyperparam_dict

        elif target_recommender is UserKNNCFRecommender:
            hyperparam_dict["topK"] = 200
            hyperparam_dict["shrink"] = 0
            hyperparam_dict["similarity"] = 'jaccard'
            hyperparam_dict["normalize"] = True

            return hyperparam_dict

        elif target_recommender is MF_BPR_Cython:
            hyperparam_dict["num_factors"] = 1
            hyperparam_dict["epochs"] = 11
            hyperparam_dict["batch_size"] = 1
            hyperparam_dict["learning_rate"] = 0.01

            return hyperparam_dict

        elif target_recommender is FunkSVD:
            hyperparam_dict["num_factors"] = 1
            hyperparam_dict["epochs"] = 30
            hyperparam_dict["reg"] = 1e-5
            hyperparam_dict["learning_rate"] = 1e-4

            return hyperparam_dict

        elif target_recommender is SLIM_BPR_Cython:
            hyperparam_dict["sgd_mode"] = 'adagrad'
            hyperparam_dict["epochs"] = 21
            hyperparam_dict["batch_size"] = 1
            hyperparam_dict["learning_rate"] = 0.1
            hyperparam_dict["topK"] = 200

            return hyperparam_dict

        elif target_recommender is SLIM_ElasticNet or target_recommender is MultiThreadSLIM_ElasticNet:
            hyperparam_dict["topK"] = 200
            hyperparam_dict["positive_only"] = True
            hyperparam_dict["l1_penalty"] = 1e-5
            hyperparam_dict["l2_penalty"] = 1e-2

            return hyperparam_dict

        elif target_recommender is P3alphaRecommender:
            hyperparam_dict["topK"] = 150
            hyperparam_dict["alpha"] = 1.3
            hyperparam_dict["normalize_similarity"] = True

            return hyperparam_dict


        elif target_recommender is RP3betaRecommender:
            hyperparam_dict["topK"] = 150
            hyperparam_dict["alpha"] = 0.9
            hyperparam_dict["beta"] = 0.6
            hyperparam_dict["normalize_similarity"] = True

            return hyperparam_dict

        print("NetflixEnhancedReader: No optimal parameters available for algorithm of class {}".format(target_recommender))

        return hyperparam_dict







    def get_model_for_rec_class(self, target_recommender, item_id_mapper = None):

        import pickle

        from KNN.item_knn_CBF import ItemKNNCBFRecommender
        from KNN.item_knn_CF import ItemKNNCFRecommender
        from KNN.user_knn_CF import UserKNNCFRecommender
        from SLIM_ElasticNet.SLIM_ElasticNet import MultiThreadSLIM_ElasticNet
        from SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet
        from GraphBased.P3alpha import P3alphaRecommender
        from GraphBased.RP3beta import RP3betaRecommender

        try:
            from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
            from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
            from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
        except ImportError:
            MF_BPR_Cython = None
            FunkSVD = None
            SLIM_BPR_Cython = None


        trainedModelFolder = "./data/" + self.DATASET_SUBFOLDER + "trained_models/"

        modelRootName = trainedModelFolder + target_recommender.RECOMMENDER_NAME + "_"

        print("NetflixEnhancedReader: If you are using a different split remember to verify the items are in the correct"
              "order, otherwise you would be using an inconsistent matrix and performance will be very poor.")

        if target_recommender is ItemKNNCFRecommender:
            W_sparse = sps.load_npz(open(modelRootName + "W_sparse.npz", "rb"))

        elif target_recommender is P3alphaRecommender:
            W_sparse = sps.load_npz(open(modelRootName + "W_sparse.npz", "rb"))

        elif target_recommender is RP3betaRecommender:
            W_sparse = sps.load_npz(open(modelRootName + "W_sparse.npz", "rb"))

        elif target_recommender is SLIM_BPR_Cython:
            W_sparse = sps.csr_matrix(sps.load_npz(open(modelRootName + "W.npz", "rb")))

        elif target_recommender is SLIM_ElasticNet or target_recommender is MultiThreadSLIM_ElasticNet:
            W_sparse = sps.load_npz(open(modelRootName + "W_sparse.npz", "rb"))

        else:
            print("NetflixEnhancedReader: No trained model available for algorithm of class {}".format(target_recommender))
            return None


        if item_id_mapper is not None:

            W_sparse = W_sparse.tocoo()
            W_sparse.col = item_id_mapper[W_sparse.col]
            W_sparse.row = item_id_mapper[W_sparse.row]

            W_sparse = W_sparse.tocsr()


        return W_sparse