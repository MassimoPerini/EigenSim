#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile


from data.DataReader import DataReader, reconcile_mapper_with_removed_tokens, removeFeatures
from data.URM_Dense_K_Cores import select_k_cores


class BookCrossingReader(DataReader):
    """
    Collected from: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

    """

    DATASET_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    DATASET_SUBFOLDER = "BookCrossing/"
    AVAILABLE_ICM = ["ICM_all", "ICM_book_crossing", "ICM_amazon"]
    DATASET_SPECIFIC_MAPPER = ["tokenToFeatureMapper_book_crossing", "tokenToFeatureMapper_amazon"]

    EDITORIAL_ICM = "ICM_amazon"


    def __init__(self, apply_k_cores = None):

        super(BookCrossingReader, self).__init__(apply_k_cores = apply_k_cores)

        print("BookCrossingReader: Ratings are in range 1-10, value -1 refers to an implicit rating")
        print("BookCrossingReader: ICM contains the author, publisher, year and tokens from the title")


    def load_from_original_file(self):
        # Load data from original

        print("BookCrossingReader: Ratings are in range 1-10, value -1 refers to an implicit rating")
        print("BookCrossingReader: ICM contains the author, publisher, year and tokens from the title")


        print("BookCrossingReader: Loading original data")

        zipFile_path =  "./data/" + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(zipFile_path + "BX-CSV-Dump.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("BookCrossingReader: Unable to fild data zip file. Downloading...")

            self.downloadFromURL(self.DATASET_URL, zipFile_path + "BX-CSV-Dump.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "BX-CSV-Dump.zip")


        URM_path = dataFile.extract("BX-Book-Ratings.csv", path=zipFile_path + "BX-CSV-Dump")
        ICM_path = dataFile.extract("BX-Books.csv", path=zipFile_path + "BX-CSV-Dump")

        self.tokenToFeatureMapper_book_crossing = {}

        print("BookCrossingReader: loading ICM")
        self.ICM_book_crossing = self._loadICM(self.tokenToFeatureMapper_book_crossing, ICM_path, separator=';', header=True, if_new_item = "add")

        self.ICM_book_crossing, _, self.tokenToFeatureMapper_book_crossing = removeFeatures(self.ICM_book_crossing, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                            reconcile_mapper = self.tokenToFeatureMapper_book_crossing)


        #############################
        ##########
        ##########      Load metadata using AmazonReviewData
        ##########      for books ASIN corresponds to ISBN

        print("BookCrossingReader: loading ICM from AmazonReviewData")

        from data.AmazonReviewData.AmazonReviewDataReader import AmazonReviewDataReader

        self.tokenToFeatureMapper_amazon = {}

        # Pass "self" object as it contains the item_id mapper already initialized with the ISBN
        self.ICM_amazon = AmazonReviewDataReader._loadMetadata(self, self.tokenToFeatureMapper_amazon, if_new = "add")

        self.ICM_amazon, _, self.tokenToFeatureMapper_amazon = removeFeatures(self.ICM_amazon, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                              reconcile_mapper=self.tokenToFeatureMapper_amazon)


        print("BookCrossingReader: loading URM")
        self.URM_all = self._loadURM(URM_path, separator=";", header = True, if_new_user = "add", if_new_item = "ignore")

        #self.URM_all, removedUsers, removedItems = removeZeroRatingRowAndCol(self.URM_all)
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)


        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        print("BookCrossingReader: Removed {} users and {} items with no interactions".format(len(removedUsers), len(removedItems)))

        ICM_filter_mask = np.ones(self.n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False

        self.ICM_book_crossing = self.ICM_book_crossing[ICM_filter_mask,:]
        self.ICM_amazon = self.ICM_amazon[ICM_filter_mask,:]


        self.ICM_all = sps.hstack([self.ICM_book_crossing, self.ICM_amazon], format='csr')

        print("BookCrossingReader: URM and ICM")
        sps.save_npz(self.data_path + "URM_all{}.npz".format(self.k_cores_name_suffix), self.URM_all)

        for icm_to_save_name in self.AVAILABLE_ICM:
            sps.save_npz(self.data_path + "{}{}.npz".format(icm_to_save_name, self.k_cores_name_suffix), self.__getattribute__(icm_to_save_name))


        self.save_mappers()

        print("BookCrossingReader: loading complete")




    def _loadURM (self, filePath, header = True, separator="::", if_new_user = "add", if_new_item = "ignore"):


        if if_new_user not in ["add", "ignore", "exception"]:
            raise ValueError("DataReader: if_new_user parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_user))

        if if_new_item not in ["add", "ignore", "exception"]:
            raise ValueError("DataReader: if_new_item parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_item))

        if if_new_user == "ignore":
            if_new_user_get_user_index = "exception"
        else:
            if_new_user_get_user_index = if_new_user

        if if_new_item == "ignore":
            if_new_item_get_item_index = "exception"
        else:
            if_new_item_get_item_index = if_new_item


        values, rows, cols = [], [], []

        fileHandle = open(filePath, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:

            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            line = line.replace('"', '')

            #print(line)

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                try:
                    user_index = self._get_user_index(line[0], if_new = if_new_user_get_user_index)
                    item_index = self._get_item_index(line[1], if_new = if_new_item_get_item_index)

                except KeyError:
                    # Go to next line
                    print("BookCrossingReader: URM contains ISBN which is not in ICM: {}. Skipping...".format(line[1]))
                    continue

                # If 0 rating is implicit
                # To avoid removin it accidentaly, set ti to -1
                rating = float(line[2])

                if rating == 0:
                    rating = -1


                rows.append(user_index)
                cols.append(item_index)
                values.append(rating)

        fileHandle.close()

        return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)



    def _loadICM(self, tokenToFeatureMapper, ICM_path, header=True, separator=',', if_new_item = "add"):

        # Pubblication Data and word in title
        from data.TagPeprocessing import tagFilterAndStemming


        values, rows, cols = [], [], []

        fileHandle = open(ICM_path, "r", encoding='latin1')
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:

                line = line.replace('"', '')
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                itemIndex = self._get_item_index(line[0], if_new = if_new_item)

                # Book Title
                featureTokenList = tagFilterAndStemming(line[1])
                # # Book author
                # featureTokenList.extend(tagFilterAndStemming(line[2]))
                # # Book year
                # featureTokenList.extend(tagFilterAndStemming(line[3]))
                # # Book publisher
                # featureTokenList.extend(tagFilterAndStemming(line[4]))

                #featureTokenList = tagFilterAndStemming(" ".join([line[1], line[2], line[3], line[4]]))

                featureTokenList.extend([line[2], line[3], line[4]])

                featureIDList = []

                for newToken in featureTokenList:

                    if newToken in tokenToFeatureMapper:
                        featureIDList.append(tokenToFeatureMapper[newToken])
                    else:
                        newID = len(tokenToFeatureMapper)
                        tokenToFeatureMapper[newToken] = newID
                        featureIDList.append(newID)

                # Rows movie ID
                # Cols features
                rows.extend([itemIndex]*len(featureIDList))
                cols.extend(featureIDList)
                values.extend([True]*len(featureIDList))


        fileHandle.close()


        return sps.csr_matrix((values, (rows, cols)), dtype=np.bool)




    def get_statistics(self):

        super(BookCrossingReader, self).get_statistics()

        n_tags = self.get_ICM().shape[1]
        n_items = self.get_ICM().shape[0]

        print("\tNumber of features: {}\n"
              "\tFeature density in ICM: {:.2E}\n".format(
              n_tags, self.get_ICM().nnz/(n_items*n_tags)))


