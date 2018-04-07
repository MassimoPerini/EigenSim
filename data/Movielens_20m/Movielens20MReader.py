#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile

from data.DataReader import DataReader, reconcile_mapper_with_removed_tokens
from data.URM_Dense_K_Cores import select_k_cores
from Recommenders.Base.Recommender_utils import reshapeSparse


class Movielens20MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens_20m/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]
    DATASET_SPECIFIC_MAPPER = ["tokenToFeatureMapper_genres", "tokenToFeatureMapper_tags"]


    def __init__(self, apply_k_cores = None):

        super(Movielens20MReader, self).__init__(apply_k_cores = apply_k_cores)



    def load_from_original_file(self):
        # Load data from original

        print("Movielens20MReader: Loading original data")

        zipFile_path = "./data/" + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens20MReader: Unable to fild data zip file. Downloading...")

            self.downloadFromURL(self.DATASET_URL, zipFile_path + "ml-20m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")


        genres_path = dataFile.extract("ml-20m/movies.csv", path=zipFile_path)
        tags_path = dataFile.extract("ml-20m/tags.csv", path=zipFile_path)
        URM_path = dataFile.extract("ml-20m/ratings.csv", path=zipFile_path)


        self.tokenToFeatureMapper_genres = {}
        self.tokenToFeatureMapper_tags = {}

        print("Movielens20MReader: loading genres")
        self.ICM_genres = self._loadICM_genres(self.tokenToFeatureMapper_genres, genres_path, header=True, separator=',', genresSeparator="|")

        print("Movielens20MReader: loading URM")
        self.URM_all = self._loadURM(URM_path, separator=",", header = True, if_new_user = "add", if_new_item = "ignore")

        print("Movielens20MReader: loading tags")
        self.ICM_tags = self._loadICM_tags(self.tokenToFeatureMapper_tags, tags_path, header=True, separator=',',
                                           if_new_user = "ignore", if_new_item = "ignore")

        self.n_items = self.ICM_genres.shape[0]

        self.ICM_genres = reshapeSparse(self.ICM_genres, (self.n_items, self.ICM_genres.shape[1]))
        self.ICM_tags = reshapeSparse(self.ICM_tags, (self.n_items, self.ICM_tags.shape[1]))


        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        ICM_filter_mask = np.ones(self.n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False

        self.ICM_genres = self.ICM_genres[ICM_filter_mask,:]
        self.ICM_tags = self.ICM_tags[ICM_filter_mask,:]


        self.ICM_all = sps.hstack([self.ICM_genres, self.ICM_tags], format='csr')



        print("Movielens20MReader: saving URM and ICM")

        sps.save_npz(self.data_path + "URM_all.npz", self.URM_all)

        for icm_to_save_name in self.AVAILABLE_ICM:
            sps.save_npz(self.data_path + "{}.npz".format(icm_to_save_name), self.__getattribute__(icm_to_save_name))

        self.save_mappers()

        print("Movielens20MReader: loading complete")




    def _loadURM (self, filePath, header = False, separator="::", if_new_user = "add", if_new_item = "ignore"):

        values, rows, cols = [], [], []

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")


                if not line[2] == "0" and not line[2] == "NaN":

                    userIndex = self._get_user_index(int(line[0]), if_new = if_new_user)
                    # Movies must be in the GENRE ICM file
                    movieIndex = self._get_item_index(int(line[1]), if_new = if_new_item)

                    rows.append(userIndex)
                    cols.append(movieIndex)
                    values.append(float(line[2]))

        fileHandle.close()

        self.n_users = len(self.user_original_ID_to_index)
        self.n_items = len(self.item_original_ID_to_index)

        shape = (self.n_users, self.n_items)

        return  sps.csr_matrix((values, (rows, cols)), shape = shape,  dtype=np.float32)




    def _loadICM_genres(self, token_to_feature_mapper, genres_path, header=True, separator=',', genresSeparator="|"):

        # Genres

        values, rows, cols = [], [], []

        fileHandle = open(genres_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # This is the first file to read, all movieIndex have to be added
                movieIndex = self._get_item_index(int(line[0]), if_new = "add")

                title = line[1]
                # In case the title contains commas, it is enclosed in "..."
                # genre list will always be the last element
                genreList = line[-1]

                genreList = genreList.split(genresSeparator)

                featureIDList = []

                for newGenre in genreList:

                    if newGenre in token_to_feature_mapper:
                        featureIDList.append(token_to_feature_mapper[newGenre])
                    else:
                        newID = len(token_to_feature_mapper)
                        token_to_feature_mapper[newGenre] = newID
                        featureIDList.append(newID)

                # Rows movie ID
                # Cols features
                rows.extend([movieIndex]*len(featureIDList))
                cols.extend(featureIDList)
                values.extend([True]*len(featureIDList))

        fileHandle.close()

        return sps.csr_matrix((values, (rows, cols)), dtype=np.bool)



    def _loadICM_tags(self, token_to_feature_mapper, tags_path, header=True, separator=',', onlyInURM = None, if_new_user = "add", if_new_item = "ignore"):

        # Tags

        from data.TagPeprocessing import tagFilterAndStemming

        values_all, rows_all, cols_all = [], [], []
        values_train, rows_train, cols_train = [], [], []

        if onlyInURM is not None:
            samplesInTrain = onlyInURM.nonzero()
            samplesInTrain = set(zip(*samplesInTrain))




        fileHandle = open(tags_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # If a movie has no genre, ignore it
                userIndex = self._get_user_index(int(line[0]), if_new = if_new_user)
                movieIndex = self._get_item_index(int(line[1]), if_new = if_new_item)

                tagList = line[2]

                # Remove non alphabetical character and split on spaces
                tagList = tagFilterAndStemming(tagList)
                tag_id_list = []

                for newTag in tagList:

                    if newTag in token_to_feature_mapper:
                        tag_id_list.append(token_to_feature_mapper[newTag])
                    else:
                        tag_id = len(token_to_feature_mapper)
                        token_to_feature_mapper[newTag] = tag_id
                        tag_id_list.append(tag_id)

                # Rows movie ID
                # Cols features
                rows_all.extend([movieIndex]*len(tag_id_list))
                cols_all.extend(tag_id_list)
                values_all.extend([1]*len(tag_id_list))

                if onlyInURM is not None:
                    if (userIndex, movieIndex) in samplesInTrain:
                        rows_train.extend([movieIndex]*len(tag_id_list))
                        cols_train.extend(tag_id_list)
                        values_train.extend([1]*len(tag_id_list))



        fileHandle.close()

        self.n_items = len(self.item_original_ID_to_index)

        shape = (self.n_items, len(token_to_feature_mapper))

        ICM_tags_all = sps.csr_matrix((values_all, (rows_all, cols_all)), dtype=np.int, shape=shape)

        if onlyInURM is not None:
            ICM_tags_train = sps.csr_matrix((values_train, (rows_train, cols_train)), dtype=np.int, shape=shape)
            return ICM_tags_all, ICM_tags_train



        return ICM_tags_all







    def get_statistics(self):

        super(Movielens20MReader, self).get_statistics()

        # n_genres = self.ICM_genres.shape[1]
        # n_tags = self.ICM_tags_train.shape[1]
        n_features = self.get_ICM().shape[1]
        n_items = self.get_ICM().shape[0]

        print("\tNumber of unique features (genres + tags): {}\n"
              "\tICM density: {:.2E}\n".format(
              n_features, self.get_ICM().nnz/(n_items*n_features)))

