#!/usr/bin/env python

# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.sparse as sps

from SLIM_BPR.SLIM_PROVA_BPR_Theano import SLIM_PROVA_BPR_Theano
from Theano.theano_bpr.bpr import BPR
from Theano.theano_bpr.utils import load_data_from_csv
from Theano.theano_bpr.utils import load_data_from_movielens


def loadCSVintoSparse (filePath):

    values, rows, cols = [], [], []

    fileHandle = open(filePath, "r")
    numCells = 0
    fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(",")

            value = line[2].replace("\n", "")

            if not value == "0" and not value == "NaN":
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(value))

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)




# Loading train data
training_data, users_to_index, items_to_index = load_data_from_movielens('http://files.grouplens.org/datasets/movielens/ml-100k/ua.base', 3)
# Loading test data
testing_data, users_to_index, items_to_index = load_data_from_movielens('http://files.grouplens.org/datasets/movielens/ml-100k/ua.test', 3, users_to_index, items_to_index)

# Initialising BPR model, 10 latent factors
bpr = BPR(10, len(users_to_index.keys()), len(items_to_index.keys()))
# Training model, 30 epochs
#bpr.train(training_data, epochs=30)
# Testing model
#print(bpr.test(testing_data))


trainCSV = "data/ml100k/train.csv"
testCSV = "data/ml100k/test.csv"

# Loading train data
train_data, users_to_index, items_to_index = load_data_from_csv(trainCSV)
# Loading test data
test_data, users_to_index, items_to_index = load_data_from_csv(testCSV, users_to_index, items_to_index)

URM_train = loadCSVintoSparse (trainCSV)
URM_test = loadCSVintoSparse (testCSV)

recommender = SLIM_PROVA_BPR_Theano(URM_train, URM_train.shape[0], URM_train.shape[1])

recommender.train(train_data, epochs=30)
# Testing model
#print(recommender.test(test_data))

print(recommender.evaluateRecommendationsParallel(URM_test))
