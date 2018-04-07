#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/01/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np

from Recommenders.Base.Recommender_utils import check_matrix



class DataSplitter(object):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    ICM_SPLIT_SUFFIX = [""]

    def __init__(self, dataReader_class, ICM_to_load = None, force_new_split = False, apply_k_cores = None):

        super(DataSplitter, self).__init__()

        self.dataReader_class = dataReader_class
        self.apply_k_cores = apply_k_cores

        # NOTE: the presence of K-core will influence the file name but not the attribute name
        if self.apply_k_cores is None or self.apply_k_cores == 1:
            self.k_cores_name_suffix = ""
        else:
            self.k_cores_name_suffix = "_{}-cores".format(self.apply_k_cores)

        if self.apply_k_cores is not None and self.apply_k_cores <= 0:
            raise ValueError("DataSplitter: apply_k_cores can only be either a positive number >= 1 or None. Provided value was '{}'".format(self.apply_k_cores))


        # If None, load all available ICM
        if ICM_to_load is None:
            self.ICM_to_load = self.dataReader_class.AVAILABLE_ICM.copy()

        elif ICM_to_load in self.dataReader_class.AVAILABLE_ICM:
            self.ICM_to_load = [ICM_to_load]

        else:
            raise ValueError("DataSplitter: ICM_to_load not among valid ICM for given dataReader_class. Available are: {}, given was {}".format(
                self.dataReader_class.AVAILABLE_ICM, ICM_to_load))


        if not force_new_split:

            try:
                self._load_split_data_and_attributes()

                self.get_statistics_URM()
                self.get_statistics_ICM()

                return

            except FileNotFoundError:

                print("DataSplitter: Split for dataset {} not found".format(dataReader_class))


        print("DataSplitter: Generating new split")

        # Call the data reader to load the data from the original data structures
        dataReader = dataReader_class(apply_k_cores = self.apply_k_cores)

        self._split_data_from_original_dataset(dataReader)
        self._load_split_data_and_attributes()


        self.get_statistics_URM()
        self.get_statistics_ICM()



    def _split_data_from_original_dataset(self, dataReader):
        raise NotImplementedError("DataSplitter: _split_data_from_original_dataset not implemented")




    def _load_split_data_and_attributes(self):
        """
        Loads all URM and ICM
        :return:
        """



        data_path = "./data/" + self.dataReader_class.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER

        for urm_to_load_name in self.dataReader_class.AVAILABLE_URM:
            setattr(self, urm_to_load_name, sps.load_npz(data_path + "{}{}.npz".format(urm_to_load_name, self.k_cores_name_suffix)))


        for icm_to_load_name in self.ICM_to_load:

            for icm_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_name_complete = "{}{}".format(icm_to_load_name, icm_suffix)

                setattr(self, ICM_name_complete, sps.load_npz(data_path + "{}{}.npz".format(ICM_name_complete, self.k_cores_name_suffix)))


        split_attributes_file = np.load(data_path + "split_attributes{}.npz".format(self.k_cores_name_suffix))

        for attribute in split_attributes_file.files:
            setattr(self, attribute, split_attributes_file[attribute])



    def split(self):
        raise NotImplementedError("DataSplitter: split not implemented")

    def get_statistics_URM(self):
        raise NotImplementedError("DataSplitter: get_statistics_URM not implemented")

    def get_statistics_ICM(self):
        raise NotImplementedError("DataSplitter: get_statistics_ICM not implemented")



    def get_URM_train(self):
        return self.URM_train.copy()

    def get_URM_test(self):
        return self.URM_test.copy()

    def get_URM_validation(self):
        return self.URM_validation.copy()


    def get_ICM_dict(self):
        """
        Returns a dict containing all the splits for the selected ICM(s)
        :return:
        """

        ICM_dict = {}


        for ICM_name in self.ICM_to_load:

            for ICM_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)
                ICM_object = getattr(self, ICM_complete_name).copy()

                ICM_dict[ICM_complete_name] = ICM_object.copy()


        return ICM_dict




    def get_split_for_specific_ICM(self, ICM_name):
        """
        Returns a dict containing all the splits for the selected ICM(s)
        OR returns only that specific ICM, if there are no splits
        :return:
        """

        if len(self.ICM_SPLIT_SUFFIX) == 1 and self.ICM_SPLIT_SUFFIX[0] == "":
            return getattr(self, ICM_name).copy()


        ICM_dict = {}

        for ICM_suffix in self.ICM_SPLIT_SUFFIX:

            ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)
            ICM_object = getattr(self, ICM_complete_name).copy()

            ICM_dict[ICM_complete_name] = ICM_object.copy()


        return ICM_dict





    def splitICM(self, dataReader):
        """
        SplitICM is required only by the cold item splitter, otherwise the original ICM is copied
        :param ICM:
        :return:
        """
        self.ICM = dataReader.get_ICM()



########################################################################################################################
##############################################
##############################################          WARM ITEMS
##############################################



class DataSplitter_Warm(DataSplitter):
    """
    This splitter performs a Holdout from the full URM splitting in train, test and validation
    """

    SPLIT_SUBFOLDER = "warm/"

    def __init__(self, dataReader_class, ICM_to_load = None, force_new_split = False, apply_k_cores = None):

        super(DataSplitter_Warm, self).__init__(dataReader_class, ICM_to_load = ICM_to_load,
                                                force_new_split = force_new_split, apply_k_cores = apply_k_cores)



    def _split_data_from_original_dataset(self, dataReader, splitProbability = list([0.6, 0.2, 0.2])):

        if sum(splitProbability) != 1.0:
            ValueError("DataSplitter: splitProbability must be a probability distribution over Train, Test and Validation. "
                       "Current value is {}".format(splitProbability))


        URM = dataReader.URM_all.tocoo()

        shape = URM.shape

        numInteractions= len(URM.data)

        split = np.random.choice([1, 2, 3], numInteractions, p=splitProbability)


        trainMask = split == 1
        self.URM_train = sps.coo_matrix((URM.data[trainMask], (URM.row[trainMask], URM.col[trainMask])), shape = shape)
        self.URM_train = self.URM_train.tocsr()

        testMask = split == 2

        self.URM_test = sps.coo_matrix((URM.data[testMask], (URM.row[testMask], URM.col[testMask])), shape = shape)
        self.URM_test = self.URM_test.tocsr()

        validationMask = split == 3

        self.URM_validation = sps.coo_matrix((URM.data[validationMask], (URM.row[validationMask], URM.col[validationMask])), shape = shape)
        self.URM_validation = self.URM_validation.tocsr()


        data_path = "./data/" + dataReader.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER

        self.n_items = self.URM_train.shape[1]
        self.n_users = self.URM_train.shape[0]

        self.item_id_mapper = np.arange(0, self.n_items, dtype=np.int)

        self.n_train_items = self.n_items
        self.n_validation_items = self.n_items
        self.n_test_items = self.n_items

        np.savez(data_path + "split_attributes{}".format(self.k_cores_name_suffix),
                 n_train_items = self.n_train_items, n_validation_items = self.n_validation_items, n_test_items = self.n_test_items,
                 n_users = self.n_users, n_items = self.n_items,
                 item_id_mapper = self.item_id_mapper)

        sps.save_npz(data_path + "URM_train{}.npz".format(self.k_cores_name_suffix), self.URM_train)
        sps.save_npz(data_path + "URM_test{}.npz".format(self.k_cores_name_suffix), self.URM_test)
        sps.save_npz(data_path + "URM_validation{}.npz".format(self.k_cores_name_suffix), self.URM_validation)



        for ICM_name in self.ICM_to_load:
            sps.save_npz(data_path + "{}{}.npz".format(ICM_name, self.k_cores_name_suffix), getattr(dataReader, ICM_name))

        print("DataSplitter: Split complete")





    def get_ICM(self):
        """
        Returns either the selected ICM or all available ICM for that dataset
        :return:
        """

        ICM_list = []

        for ICM_name in self.ICM_to_load:

            ICM_list.append(getattr(self, ICM_name))

        return ICM_list




    def get_statistics_URM(self):

        # This avoids the fixed bit representation of numpy preventing
        # an overflow when computing the product
        n_items = int(self.n_items)
        n_users = int(self.n_users)

        print("DataSplitter_Warm for DataReader: {}\n"
              "\t Num items: {}\n"
              "\t Num users: {}\n".format(self.dataReader_class, n_items, n_users))


        n_global_interactions = 0

        for URM_name in self.dataReader_class.AVAILABLE_URM:

            URM_object = getattr(self, URM_name)
            n_global_interactions += URM_object.nnz



        for URM_name in self.dataReader_class.AVAILABLE_URM:

            URM_object = getattr(self, URM_name)

            print("\t Statistics for {}: n_interactions {} ( {:.2f}%), density: {:.2E}".format(
                URM_name, URM_object.nnz, URM_object.nnz/n_global_interactions*100, URM_object.nnz/(int(n_items)*int(n_users))
            ))

        print("\n")



    def get_statistics_ICM(self):

        for ICM_name in self.ICM_to_load:

            ICM_object = getattr(self, ICM_name)
            n_items = ICM_object.shape[0]
            n_features = ICM_object.shape[1]

            print("\t Statistics for {}: n_features {}, feature occurrences {}, density: {:.2E}".format(
                ICM_name, n_features, ICM_object.nnz, ICM_object.nnz/(int(n_items)*int(n_features))
            ))

        print("\n")





########################################################################################################################
##############################################
##############################################          COLD ITEMS - WARM VALIDATION
##############################################



class DataSplitter_ColdItems_WarmValidation(DataSplitter):
    """
    This splitter creates a cold item split. Given the quota of samples in the test set, a number of items is randomly sampled
    in such a way to create a split with enough interactions.
    The URM validation is a holdout of the warm part
    The ICM is partitioned in ICM_warm, containin the items in the warm part, and ICM_cold containing only cold items
    """

    SPLIT_SUBFOLDER = "coldItems_warmValidation/"
    ICM_SPLIT_SUFFIX = ["_warm", "_global"]



    def __init__(self, dataReader_class, ICM_to_load = None, force_new_split = False, apply_k_cores = None):

        super(DataSplitter_ColdItems_WarmValidation, self).__init__(dataReader_class, ICM_to_load = ICM_to_load,
                                                                    force_new_split = force_new_split, apply_k_cores = apply_k_cores)



    def selectColdItems(self, URM, splitProbability_test, minItemsPercentage):

        numGlobalInteractions = URM.nnz
        current_matrix_n_items = URM.shape[1]


        numInteractionsPerItem = np.array(URM.sum(axis=0)).ravel()


        # Select cold items

        terminate = False

        cold_items = set()
        cold_items_interactions = 0

        while not terminate:

            candidate_item = np.random.randint(0, current_matrix_n_items)

            if candidate_item not in cold_items:
                cold_items.add(candidate_item)
                cold_items_interactions += numInteractionsPerItem[candidate_item]

            if cold_items_interactions >= splitProbability_test*numGlobalInteractions and\
                    len(cold_items)>= minItemsPercentage*current_matrix_n_items:
                terminate = True


        return cold_items




    def splitColdItems(self, URM, splitProbability_test, minItemsPercentage):


        cold_items = self.selectColdItems(URM, splitProbability_test, minItemsPercentage)
        current_matrix_n_items = URM.shape[1]


        cold_items = np.array(list(cold_items))
        n_cold_items = len(cold_items)
        n_warm_items = current_matrix_n_items - n_cold_items


        # Redefine indices in such a way that the matrix contains warm items before cold item
        warm_item_mask = np.in1d(np.arange(0, current_matrix_n_items), cold_items, invert=True)

        item_id_mapper = np.zeros(current_matrix_n_items, dtype=np.int)

        item_id_mapper[warm_item_mask] = np.arange(0, n_warm_items, dtype=np.int)
        item_id_mapper[cold_items] = np.arange(n_warm_items, current_matrix_n_items, dtype=np.int)

        # Rearrange URM columns accordingly
        URM = URM.tocoo()

        URM.col = item_id_mapper[URM.col]

        URM = check_matrix(URM, 'csc')

        # URM_cold_items has zero values for all warm items
        URM_cold_items = URM.copy()

        for warm_item in range(0, n_warm_items):
            URM_cold_items.data[URM_cold_items.indptr[warm_item]:URM_cold_items.indptr[warm_item+1]] = 0

        URM_cold_items.eliminate_zeros()



        # URM_warm_items has zero values for all cold items
        URM_warm_items = URM.copy()

        for cold_item in range(n_warm_items, current_matrix_n_items):
            URM_warm_items.data[URM_warm_items.indptr[cold_item]:URM_warm_items.indptr[cold_item+1]] = 0

        URM_warm_items.eliminate_zeros()



        return URM_cold_items, URM_warm_items, cold_items, item_id_mapper






    def _split_data_from_original_dataset(self, dataReader, splitProbability = list([0.6, 0.2, 0.2]), minItemsPercentage = 0.20):

        if sum(splitProbability) != 1.0:
            ValueError("DataSplitter: splitProbability must be a probability distribution over Train, Test and Validation. "
                       "Current value is {}".format(splitProbability))


        splitProbability_train = splitProbability[0]
        splitProbability_test = splitProbability[1]
        splitProbability_validation = splitProbability[2]



        URM = dataReader.URM_all.tocoo()


        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]

        numGlobalInteractions = URM.nnz


        self.URM_test, URM_warm_items, self.cold_items, self.item_id_mapper =\
            self.splitColdItems(URM, splitProbability_test, minItemsPercentage)

        self.cold_items = np.array(list(self.cold_items))
        self.n_test_items = len(self.cold_items)
        self.n_warm_items = self.n_items - self.n_test_items
        self.n_train_items = self.n_warm_items
        self.n_validation_items = 0



        # Build holdout URM train and validation
        # Split probability will be adjusted to take into account the reduced matrix size
        numWarmInteractions = URM_warm_items.nnz

        splitProbability_test = self.URM_test.nnz / numGlobalInteractions
        splitProbability_train = splitProbability_train / (1 - splitProbability_test)
        splitProbability_validation = 1 - splitProbability_train

        split = np.random.choice([1, 2], numWarmInteractions , p=[splitProbability_train, splitProbability_validation])

        URM_warm_items = URM_warm_items.tocoo()

        trainMask = split == 1
        self.URM_train = sps.coo_matrix((URM_warm_items.data[trainMask], (URM_warm_items.row[trainMask], URM_warm_items.col[trainMask])),
                                        shape = (self.n_users, self.n_items))
        self.URM_train = self.URM_train.tocsr()

        validationMask = split == 2
        self.URM_validation = sps.coo_matrix((URM_warm_items.data[validationMask], (URM_warm_items.row[validationMask], URM_warm_items.col[validationMask])),
                                        shape = (self.n_users, self.n_items))


        self.URM_validation = self.URM_validation.tocsr()

        data_path = "./data/" + dataReader.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER

        np.savez(data_path + "split_attributes{}".format(self.k_cores_name_suffix),
                 n_train_items = self.n_warm_items, n_validation_items = self.n_warm_items, n_test_items = self.n_test_items,
                 n_users = self.n_users, n_items = self.n_items,
                 item_id_mapper = self.item_id_mapper)

        sps.save_npz(data_path + "URM_train{}.npz".format(self.k_cores_name_suffix), self.URM_train)
        sps.save_npz(data_path + "URM_test{}.npz".format(self.k_cores_name_suffix), self.URM_test)
        sps.save_npz(data_path + "URM_validation{}.npz".format(self.k_cores_name_suffix), self.URM_validation)

        print("DataSplitter: Split complete")


        self.splitICM(dataReader)




    def splitICM(self, dataReader):

        for ICM_name in self.ICM_to_load:

            ICM = getattr(dataReader, ICM_name).copy().tocoo()
            self.n_features = ICM.shape[1]

            ICM.row = self.item_id_mapper[ICM.row]

            ICM = check_matrix(ICM, 'csr')

            # ICM_warm has zero values for all cold items
            self.ICM_warm = ICM.copy()

            for warm_item in range(self.n_warm_items, self.n_items):
                self.ICM_warm.data[self.ICM_warm.indptr[warm_item]:self.ICM_warm.indptr[warm_item+1]] = 0

            self.ICM_warm.eliminate_zeros()

            # ICM_global contains all data
            self.ICM_global = ICM.copy()

            self.ICM_global.eliminate_zeros()

            # ICM contains war mdata
            # ICM cold Items contains all data
            data_path = "./data/" + dataReader.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER

            # Build the name as a combination of ICM_name + suffix
            # Ensure the object has a copy of such object in self
            for ICM_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)
                ICM_object = getattr(self, "ICM{}".format(ICM_suffix)).copy()

                setattr(self, ICM_complete_name, ICM_object)

                sps.save_npz(data_path + "{}{}.npz".format(ICM_complete_name, self.k_cores_name_suffix), ICM_object)




    def get_train_items(self):
        return list(range(0, self.n_train_items))

    def get_validation_items(self):
        return list(range(0, self.n_train_items))

    def get_test_items(self):
        return list(range(self.n_items-self.n_test_items, self.n_items))




    def get_ICM_cold(self):
        """
        Returns either the selected ICM or all available ICM for that dataset
        :return:
        """

        ICM_list = []

        for ICM_name in self.ICM_to_load:

            ICM_complete_name = "{}_global".format(ICM_name)
            ICM_list.append(getattr(self, ICM_complete_name))

        return ICM_list



    def get_ICM_warm(self):
        """
        Returns either the selected ICM or all available ICM for that dataset
        :return:
        """

        ICM_list = []

        for ICM_name in self.ICM_to_load:

            ICM_complete_name = "{}_warm".format(ICM_name)
            ICM_list.append(getattr(self, ICM_complete_name))

        return ICM_list






    def get_statistics_URM(self):

        n_interactions_warm = self.URM_train.nnz + self.URM_validation.nnz
        n_interactions_cold = self.URM_test.nnz

        n_interactions_all = n_interactions_warm + n_interactions_cold

        n_users = self.URM_train.shape[0]


        print("DataSplitter_Cold for DataReader: {}\n"
              "\tNumber of items: {}\n"
              "\t\tNumber of cold items: {} ( {:.2f}%)\n"
              "\t\tNumber of warm items: {} ( {:.2f}%)\n".format(
                self.dataReader_class,
                self.n_items,
                self.n_test_items,
                self.n_test_items / self.n_items * 100,
                self.n_train_items,
                self.n_train_items / self.n_items * 100))


        print("\tNumber of users: {}\n"
              "\tNumber of interactions in global train: {}, density {:.2E}\n"
              "\t\tNumber of interactions in cold URM: {} ( {:.2f}%), density {:.2E}\n"
              "\t\tNumber of interactions in warm URM: {} ( {:.2f}%), density {:.2E}\n".format(
                n_users,
                n_interactions_all,
                n_interactions_all/(int(self.n_items) * int(n_users)),
                n_interactions_cold,
                n_interactions_cold / n_interactions_all *100,
                n_interactions_cold/(int(self.n_test_items) * int(n_users)),
                n_interactions_warm,
                n_interactions_warm / n_interactions_all *100,
                n_interactions_warm/(int(self.n_train_items) * int(n_users))))


    def get_statistics_ICM(self):


        for ICM_name in self.ICM_to_load:

            print("\n\t Statistics for {}:".format(ICM_name))

            for ICM_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)

                ICM_object = getattr(self, ICM_complete_name)
                n_items = ICM_object.shape[0]
                n_features = ICM_object.shape[1]

                print("\t\t Split: {}, n_features {}, feature occurrences {}, density: {:.2E}".format(
                    ICM_suffix, n_features, ICM_object.nnz, ICM_object.nnz/(int(n_features) * int(n_items))
                ))

        print("\n")











########################################################################################################################
##############################################
##############################################          COLD ITEMS - COLD VALIDATION
##############################################



from Recommenders.Base.Recommender_utils import reshapeSparse


class DataSplitter_ColdItems_ColdValidation(DataSplitter):
    """
    This splitter creates a cold item split. Given the quota of samples in the test set, a number of items is randomly sampled
    in such a way to create a split with enough interactions.
    The URM validation and ICM validation are both cold start
    The ICM is partitioned in ICM_warm, containin the items in the warm part, and ICM_cold containing only cold items
    """

    SPLIT_SUBFOLDER = "coldItems_coldValidation/"
    ICM_SPLIT_SUFFIX = ["_train", "_validation", "_test"]


    def __init__(self, dataReader_class, ICM_to_load = None, force_new_split = False, apply_k_cores = None):

        super(DataSplitter_ColdItems_ColdValidation, self).__init__(dataReader_class, ICM_to_load = ICM_to_load,
                                                                    force_new_split = force_new_split, apply_k_cores = apply_k_cores)


    def selectColdItems(self, URM, splitProbability_test, minItemsPercentage):

        numGlobalInteractions = URM.nnz
        current_matrix_n_items = URM.shape[1]


        numInteractionsPerItem = np.array(URM.sum(axis=0)).ravel()


        # Select cold items

        terminate = False

        cold_items = set()
        cold_items_interactions = 0

        while not terminate:

            candidate_item = np.random.randint(0, current_matrix_n_items)

            if candidate_item not in cold_items:
                cold_items.add(candidate_item)
                cold_items_interactions += numInteractionsPerItem[candidate_item]

            if cold_items_interactions >= splitProbability_test*numGlobalInteractions and\
                    len(cold_items)>= minItemsPercentage*current_matrix_n_items:
                terminate = True


        return cold_items




    def splitColdItems(self, URM, splitProbability_test, minItemsPercentage):


        cold_items = self.selectColdItems(URM, splitProbability_test, minItemsPercentage)
        current_matrix_n_items = URM.shape[1]


        cold_items = np.array(list(cold_items))
        n_cold_items = len(cold_items)
        n_warm_items = current_matrix_n_items - n_cold_items


        # Redefine indices in such a way that the matrix contains warm items before cold item
        warm_item_mask = np.in1d(np.arange(0, current_matrix_n_items), cold_items, invert=True)

        item_id_mapper = np.zeros(current_matrix_n_items, dtype=np.int)

        item_id_mapper[warm_item_mask] = np.arange(0, n_warm_items, dtype=np.int)
        item_id_mapper[cold_items] = np.arange(n_warm_items, current_matrix_n_items, dtype=np.int)

        # Rearrange URM columns accordingly
        URM = URM.tocoo()

        URM.col = item_id_mapper[URM.col]

        URM = check_matrix(URM, 'csc')

        # URM_cold_items has zero values for all warm items
        URM_cold_items = URM.copy()

        for warm_item in range(0, n_warm_items):
            URM_cold_items.data[URM_cold_items.indptr[warm_item]:URM_cold_items.indptr[warm_item+1]] = 0

        URM_cold_items.eliminate_zeros()



        # URM_warm_items has zero values for all cold items
        URM_warm_items = URM.copy()

        for cold_item in range(n_warm_items, current_matrix_n_items):
            URM_warm_items.data[URM_warm_items.indptr[cold_item]:URM_warm_items.indptr[cold_item+1]] = 0

        URM_warm_items.eliminate_zeros()



        return URM_cold_items, URM_warm_items, cold_items, item_id_mapper






    def _split_data_from_original_dataset(self, dataReader, splitProbability = list([0.6, 0.2, 0.2]), minItemsPercentage = 0.20):

        if sum(splitProbability) != 1.0:
            ValueError("DataSplitter: splitProbability must be a probability distribution over Train, Test and Validation. "
                       "Current value is {}".format(splitProbability))


        splitProbability_train = splitProbability[0]
        splitProbability_test = splitProbability[1]
        splitProbability_validation = splitProbability[2]



        URM = dataReader.URM_all.tocoo()


        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]

        numGlobalInteractions = URM.nnz



        self.URM_test, URM_warm_items, self.cold_items, self.item_id_mapper =\
            self.splitColdItems(URM, splitProbability_test, minItemsPercentage)


        # Update probabilities to take into account the reduced data size
        splitProbability_test = self.URM_test.nnz / numGlobalInteractions
        splitProbability_train = splitProbability_train / (1 - splitProbability_test)
        splitProbability_validation = 1 - splitProbability_train

        minItemsPercentage = minItemsPercentage/(1 - minItemsPercentage)


        self.URM_validation, self.URM_train, self.validation_items, validation_item_id_mapper =\
            self.splitColdItems(URM_warm_items[:,:self.n_items-len(self.cold_items)], splitProbability_validation, minItemsPercentage)

        # Update mapper with all validaton items
        train_or_validation_item_mask = np.in1d(np.arange(0, self.n_items), self.cold_items, invert=True)
        self.item_id_mapper[train_or_validation_item_mask] = validation_item_id_mapper



        # Ensure matrices have correct size
        correctShape = (self.n_users, self.n_items)

        self.n_train_items = self.n_items - len(self.cold_items) - len(self.validation_items)
        self.n_validation_items = len(self.validation_items)
        self.n_test_items = len(self.cold_items)

        self.URM_train = reshapeSparse(self.URM_train, correctShape)
        self.URM_validation = reshapeSparse(self.URM_validation, correctShape)


        self.URM_train = check_matrix(self.URM_train, 'csr')
        self.URM_test = check_matrix(self.URM_test, 'csr')
        self.URM_validation = check_matrix(self.URM_validation, 'csr')


        data_path = "./data/" + dataReader.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER


        np.savez(data_path + "split_attributes{}".format(self.k_cores_name_suffix),
                 n_train_items = self.n_train_items, n_validation_items = self.n_validation_items, n_test_items = self.n_test_items,
                 n_users = self.n_users, n_items = self.n_items,
                 item_id_mapper = self.item_id_mapper)

        sps.save_npz(data_path + "URM_train{}.npz".format(self.k_cores_name_suffix), self.URM_train)
        sps.save_npz(data_path + "URM_test{}.npz".format(self.k_cores_name_suffix), self.URM_test)
        sps.save_npz(data_path + "URM_validation{}.npz".format(self.k_cores_name_suffix), self.URM_validation)

        print("DataSplitter: Split complete")

        self.get_statistics_URM()
        self.splitICM(dataReader)




    def splitICM(self, dataReader):

        for ICM_name in self.ICM_to_load:

            ICM = getattr(dataReader, ICM_name).copy().tocoo()
            self.n_features = ICM.shape[1]

            ICM.row = self.item_id_mapper[ICM.row]

            ICM = check_matrix(ICM, 'csr')


            self.ICM_train = ICM.copy()
            self.ICM_validation = ICM.copy()

            # ICM_train has zero values for all validation and test items
            for cold_item in range(self.n_train_items, self.n_items):
                self.ICM_train.data[self.ICM_train.indptr[cold_item]:self.ICM_train.indptr[cold_item+1]] = 0

            self.ICM_train.eliminate_zeros()



            # ICM_validation has zero values for all test items
            for warm_item in range(self.n_items - len(self.cold_items), self.n_items):
                self.ICM_validation.data[self.ICM_validation.indptr[warm_item]:self.ICM_validation.indptr[warm_item+1]] = 0

            self.ICM_validation.eliminate_zeros()


            # ICM_test contains all data
            self.ICM_test = ICM.copy()
            self.ICM_test.eliminate_zeros()

            # ICM contains war mdata
            # ICM cold Items contains all data
            data_path = "./data/" + dataReader.DATASET_SUBFOLDER + self.SPLIT_SUBFOLDER

            # Build the name as a combination of ICM_name + suffix
            # Ensure the object has a copy of such object in self
            for ICM_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)
                ICM_object = getattr(self, "ICM{}".format(ICM_suffix)).copy()

                setattr(self, ICM_complete_name, ICM_object)

                sps.save_npz(data_path + "{}{}.npz".format(ICM_complete_name, self.k_cores_name_suffix), ICM_object)




    def get_train_items(self):
        return list(range(0, self.n_train_items))

    def get_validation_items(self):
        return list(range(self.n_train_items, self.n_train_items + self.n_validation_items))

    def get_test_items(self):
        return list(range(self.n_items-self.n_test_items, self.n_items))





    def get_statistics_URM(self):

        print("DataSplitter_ColdItems_ColdValidation for DataReader: {}\n"
              "\tNumber of items: {}\n"
              "\t\tNumber of train items: {} ( {:.2f}%)\n"
              "\t\tNumber of validation items: {} ( {:.2f}%)\n"
              "\t\tNumber of test items: {} ( {:.2f}%)\n"
              "\t Number of users: {}\n".format(
            self.dataReader_class,
            self.n_items,
            self.n_train_items, self.n_train_items / self.n_items * 100,
            self.n_validation_items, self.n_validation_items / self.n_items * 100,
            self.n_test_items, self.n_test_items / self.n_items * 100,
            self.n_users))



        n_global_interactions = 0

        for URM_name in self.dataReader_class.AVAILABLE_URM:

            URM_object = getattr(self, URM_name)
            n_global_interactions += URM_object.nnz



        for URM_name in self.dataReader_class.AVAILABLE_URM:

            URM_object = getattr(self, URM_name)

            print("\t Statistics for {}: n_interactions {} ( {:.2f}%), density: {:.2E}".format(
                URM_name, URM_object.nnz, URM_object.nnz/n_global_interactions*100, URM_object.nnz/(int(self.n_items) * int(self.n_users))
            ))

        print("\n")



    def get_statistics_ICM(self):

        print("DataSplitter_ColdItems_ColdValidation: ICM_train contains only train items, "
              "ICM_validation contains train and validation items, ICM_test contains all items")

        for ICM_name in self.ICM_to_load:

            print("\n\t Statistics for {}:".format(ICM_name))

            for ICM_suffix in self.ICM_SPLIT_SUFFIX:

                ICM_complete_name = "{}{}".format(ICM_name, ICM_suffix)

                ICM_object = getattr(self, ICM_complete_name)
                n_items = ICM_object.shape[0]
                n_features = ICM_object.shape[1]

                print("\t\t Split: {}, n_features {}, feature occurrences {}, density: {:.2E}".format(
                    ICM_suffix, n_features, ICM_object.nnz, ICM_object.nnz/(n_items * n_features)
                ))

        print("\n")
