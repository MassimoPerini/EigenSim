#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/01/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse import linalg

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix

from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from Base.Recommender_utils import similarityMatrixTopK

from GraphBased.P3alpha import P3alphaRecommender
import time, sys
import os, subprocess

class RP3betaRecommender_ML(Recommender, Similarity_Matrix_Recommender):
    """ RP3beta_LSQ recommender """

    #python compileCython.py RP3beta_Cython_epoch.pyx build_ext --inplace

    def __init__(self, URM_train, recompile_cython = False):
        super(RP3betaRecommender_ML, self).__init__()

        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.sparse_weights = True


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def __str__(self):
        return "RP3beta_LSQ(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)



    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/GraphBased/Cython"
        fileToCompile_list = ['RP3beta_Cython_epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        #python compileCython.py RP3beta_Cython_epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "RP3beta_Cython_epoch.pyx"])




    def fit(self, alpha=1., min_rating=0, topK=100, implicit=False, normalize_similarity=True,
            epochs = 30, learn_rate = 1e-2, useAdaGrad=True, objective = "RMSE"):

        self.alpha = alpha
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        # Compute probabilty matrix
        p3alpha = P3alphaRecommender(self.URM_train)
        p3alpha.fit(alpha = self.alpha, min_rating=min_rating, implicit=implicit,
                    normalize_similarity=normalize_similarity, topK=topK*10)

        self.W_sparse = p3alpha.W_sparse


        from GraphBased.Cython.RP3beta_Cython_epoch import RP3beta_ML_Cython

        cython_model = RP3beta_ML_Cython(self.URM_train, self.W_sparse)

        itemsDegree = cython_model.fit(epochs = epochs, learn_rate = learn_rate,
                                       useAdaGrad=useAdaGrad, objective = objective)



        for item_id in range(self.URM_train.shape[1]):
            self.W_sparse[item_id,:] = self.W_sparse[item_id,:].multiply(itemsDegree)


        # CSR works faster for testing
        self.W_sparse = check_matrix(self.W_sparse, 'csr')
        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=topK)

        self.URM_train = check_matrix(self.URM_train, 'csr')
