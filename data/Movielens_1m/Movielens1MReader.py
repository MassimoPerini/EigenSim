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


def loadCSVintoSparse (filePath, header = False, separator="::"):

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
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(line[2]))

    fileHandle.close()

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)




class Movielens1MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens_1m/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self, apply_k_cores = None):

        super(Movielens1MReader, self).__init__(apply_k_cores = apply_k_cores)


    def load_from_original_file(self):
        # Load data from original

        print("Movielens1MReader: Loading original data")

        zipFile_path =  "./data/" + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens1MReader: Unable to fild data zip file. Downloading...")

            self.downloadFromURL(self.DATASET_URL, zipFile_path + "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path)

        self.URM_all = loadCSVintoSparse(URM_path, separator="::")
        #self.URM_all, removedUsers, removedItems = removeZeroRatingRowAndCol(self.URM_all)
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)


        print("Movielens1MReader: saving URM_train and URM_test")
        sps.save_npz(self.data_path + "URM_all.npz", self.URM_all)

        self.save_mappers()

        print("Movielens1MReader: loading complete")

