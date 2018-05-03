#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile


from data.DataReader import DataReader, reconcile_mapper_with_removed_tokens
from data.URM_Dense_K_Cores import select_k_cores



class NetflixPrizeReader(DataReader):


    DATASET_SUBFOLDER = "NetflixPrize/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self, apply_k_cores = None):

        super(NetflixPrizeReader, self).__init__(apply_k_cores = apply_k_cores)




    def load_from_original_file(self):

        self.dataSubfolder =  "./data/" + self.DATASET_SUBFOLDER

        try:

            self.dataFile = zipfile.ZipFile(self.dataSubfolder + "netflix-prize-data.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("NetflixPrizeReader: Unable to fild data zip file. Downloading...")






        self.URM_all = self._loadUserInteractions()

        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)


        print("NetflixPrizeReader: saving URM_train and URM_test")
        sps.save_npz(self.data_path + "URM_all.npz", self.URM_all)

        self.save_mappers()

        print("NetflixPrizeReader: loading complete")






    def _loadUserInteractions(self):


        # Use array as for datasets this big lists would require more than 10GB of RAM
        dataBlock = 10000000

        values = np.zeros(dataBlock)
        rows = np.zeros(dataBlock)
        cols = np.zeros(dataBlock)

        numCells = 0

        for current_split in [1, 2, 3, 4]:

            current_split_path = self.dataFile.extract("combined_data_{}.txt".format(current_split), path=self.dataSubfolder + "netflix-prize-data/")

            fileHandle = open(current_split_path, "r")

            print("NetflixPrizeReader: loading split {}".format(current_split))

            currentMovie_id = None

            for line in fileHandle:


                if numCells % 1000000 == 0 and numCells!=0:
                    print("Processed {} cells".format(numCells))

                if (len(line)) > 1:

                    line_split = line.split(",")

                    # If line has 3 components, it is a 'user_id,rating,date' row
                    if len(line_split) == 3 and currentMovie_id!= None:

                        if numCells == len(rows):
                            rows = np.concatenate((rows, np.zeros(dataBlock)))
                            cols = np.concatenate((cols, np.zeros(dataBlock)))
                            values = np.concatenate((values, np.zeros(dataBlock)))



                        userIndex = self._get_user_index(line_split[0], if_new = "add")

                        rows[numCells] = userIndex
                        cols[numCells] = currentMovie_id
                        values[numCells] = float(line_split[1])

                        numCells += 1

                    # If line has 1 component, it MIGHT be a 'item_id:' row
                    elif len(line_split) == 1:
                        line_split = line.split(":")

                        # Confirm it is a 'item_id:' row
                        if len(line_split) == 2:
                            currentMovie_id = int(line_split[0])
                            currentMovie_id = self._get_item_index(currentMovie_id, if_new = "add")

                        else:
                            print("Unexpected row: '{}'".format(line))

                    else:
                        print("Unexpected row: '{}'".format(line))


            fileHandle.close()


        return  sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), dtype=np.float32)

