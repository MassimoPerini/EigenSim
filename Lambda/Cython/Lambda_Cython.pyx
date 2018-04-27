#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
from cpython.array cimport array, clone
cimport numpy as np
import time
import sys
import scipy.sparse.linalg

from Base.Recommender_utils import similarityMatrixTopK

from cpython.array cimport array, clone
from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from libcpp.map cimport map
from cython.operator cimport dereference as deref #C++ map
from cython.operator cimport preincrement as inc

# N.B. C++ code

#import seeds (not randomize BPR)
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef class Lambda_BPR_Cython_Epoch:
    cdef int n_users
    cdef int n_items
    cdef int numPositiveIteractions
    cdef int useAdaGrad, rmsprop
    cdef float learning_rate
    cdef long[:] eligibleUsers
    cdef long numEligibleUsers
    cdef int batch_size

    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser
    cdef double lambda_2 #coefficiente di regolarizzazione

    cdef int[:] URM_mask_indices, URM_mask_indptr
    cdef int[:] URM_mask_transp_indices, URM_mask_transp_indptr #faster access to rows index with 1 of certain item (col)
    cdef double[:] lambda_learning # lambda learned
    cdef float[:,:] pseudoInv #pseudoinverse
    cdef int enablePseudoInv, force_positive

    cdef float[:,:] SVD_U, SVD_Vh
    cdef float[:] SVD_s
    cdef int SVD_latent_factors, low_ram

    cdef double [:] sgd_cache #cache adagrad

    cdef S_sparse
    cdef URM_train

    def __init__(self, URM_mask, URM_train, eligibleUsers, rcond = 0.1, k=10,
                 learning_rate = 0.05, lambda_2=0.0002,
                 batch_size = 1, sgd_mode='sgd', enablePseudoInv = False, initialize="zero",
                 low_ram = True, force_positive = True):

        super(Lambda_BPR_Cython_Epoch, self).__init__()


        self.URM_train = check_matrix(URM_train, 'csr')
        URM_mask = check_matrix(URM_mask, 'csr')

        self.numPositiveIteractions = int(URM_mask.nnz)
        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.lambda_2=lambda_2
        self.URM_mask_indices = URM_mask.indices
        self.URM_mask_indptr = URM_mask.indptr

        URM_transposed = URM_mask.transpose()
        URM_transposed = check_matrix(URM_transposed, 'csr')

        self.URM_mask_transp_indices = URM_transposed.indices
        self.URM_mask_transp_indptr = URM_transposed.indptr
        #typr of initialization
        if initialize == "zero":
            self.lambda_learning = np.zeros(self.n_users) #init the values of the
        elif initialize == "one":
            self.lambda_learning = np.ones(self.n_users)
        elif initialize == "random":
            self.lambda_learning = np.random.normal(0.0, 0.01, (self.n_users)).astype(np.float64)
        else:
            raise ValueError(
                "init not valid. Values: zero, one, random")

        self.enablePseudoInv = enablePseudoInv
        self.low_ram = low_ram
        self.force_positive = force_positive

        if enablePseudoInv:

            if low_ram:
                self.SVD_U, self.SVD_s, self.SVD_Vh = scipy.sparse.linalg.svds(self.URM_train, k=k)
                self.SVD_latent_factors = self.SVD_U.shape[1]
            else:
                self.URM_train.astype(np.float32)
                self.pseudoInv = np.linalg.pinv(self.URM_train.todense(), rcond = rcond).astype(np.float32, copy=False)



        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache = np.zeros((self.n_users), dtype=float)
        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD not valid.")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eligibleUsers = eligibleUsers
        self.numEligibleUsers = len(eligibleUsers)



    cdef int[:] getSeenItemsOfUser(self, long index): #given an user, get the items he likes
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]

    cdef int[:] getUsersSeenItem(self, long index): # given an item, get the users who like that item
        return self.URM_mask_transp_indices[self.URM_mask_transp_indptr[index]:self.URM_mask_transp_indptr[index + 1]]

    cpdef transpose_batch (self):
        cdef long [:] batch_pos_items = np.zeros(self.batch_size, dtype=int)
        cdef long [:] batch_neg_items = np.zeros(self.batch_size, dtype=int)
        cdef map [long, long] involved_users #C++ map
        cdef map [long, long] sampled_items # C++ map
        cdef long index, index2, currentUser = 0, sampled_user = 0, how_many_items_liked_both = 0
        cdef BPR_sample sample
        cdef long[:] batch_users = np.zeros(self.batch_size, dtype=int)
        cdef int[:] users_saw_sampled_item, user_profile, usersPosItem, usersNegItem
        cdef Sparse_Matrix_Tree_CSR batch_matrix
        cdef map[long,long].iterator itr
        cdef long[:,:] dense_batch_matrix
        cdef double x_uij=0, deriv_xuij=0, gradient


        involved_users.clear()
        sampled_items.clear()
        #esegui tutti i campionamenti e tieni traccia degli utenti che devono essere inseriti nella matrice (utenti coinvolti)
        for index in range(self.batch_size):
            # campiona e salva i campionamenti
            sample = self.sampleBatch_Cython()
            batch_users[index] = sample.user            #questi 3 array contengono user, elementi pos e elementi neg. campionati
            batch_pos_items[index] = sample.pos_item
            batch_neg_items[index] = sample.neg_item

            involved_users[sample.user] = sample.user        #questa mappa contiene tutti gli user che sono richiesti per l'elaborazione futura

            if sampled_items.count(sample.pos_item) == 0: # cerco se nella mappa degli item campionati è gia' stato campionato lo stesso elemento
                                                                        # (se sì non ha senso re-inserire gli utenti a cui piace l'item nell'altra mappa visto che sono stati inseriti in precedenza)
                sampled_items[sample.pos_item] = sample.pos_item

                users_saw_sampled_item = self.getUsersSeenItem(sample.pos_item)
                for index2 in range(len(users_saw_sampled_item)):            #quindi anche quelli che hanno visto l'elemento positivo
                    involved_users[users_saw_sampled_item[index2]] = users_saw_sampled_item[index2]

            if sampled_items.count(sample.neg_item) == 0: # uguale per il campione "negativo"
                sampled_items[sample.neg_item] = sample.neg_item

                users_saw_sampled_item = self.getUsersSeenItem(sample.neg_item)
                for index2 in range(len(users_saw_sampled_item)):            #e quelli che hanno visto l'elemento negativo
                    involved_users[users_saw_sampled_item[index2]] = users_saw_sampled_item[index2]

                #crea la matrice sparsa
        batch_matrix = Sparse_Matrix_Tree_CSR(self.n_users, self.n_items)
        index = 0

        #print("Step 2")
        #inserisco i profili nella matrice (forse è più veloce usare direttamente scipy o altro)
        itr=involved_users.begin()
        while itr!=involved_users.end():         #scorro gli user della mappa e copio gli elementi che piacciono ad ogni user nella matrice
            currentUser = deref(itr).first #un utente
            user_profile = self.getSeenItemsOfUser(currentUser) #elementi che piacciono all'utente
            for index2 in range(len(user_profile)):   #copia quello che piace all'user nella matrice
                batch_matrix.add_value(currentUser, user_profile[index2], 1)
            inc(itr)

        #calcolo il prodotto con la trasposta

        #print("Step 3")

        scipy_tmp = batch_matrix.get_scipy_csr()
        scipy_tmp = scipy_tmp.astype(np.long)
        scipy_tmp = scipy_tmp.dot(scipy_tmp.T)
        #print("done")
        dense_batch_matrix = scipy_tmp.todense() # rendo densa la matrice per andare più veloce negli accessi. POTREBBE CREARE PROBLEMI con grandi campionamenti su grandi matrici
        del(scipy_tmp)

        #print("iterating... ",len(batch_users))
        for index in range(len(batch_users)):#per ogni campione calcolo e applico il gradiente...
            #if index%50000 == 0:
                #print("computing: ", index ," of ", len(batch_users))
            x_uij = 0
            sampled_user = batch_users[index]       #prendo l'utente del campione
            deriv_xuij = len(self.getSeenItemsOfUser(sampled_user))

            usersPosItem = self.getUsersSeenItem(batch_pos_items[index]) #users a cui piace l'item che piace all'utente
            for index2 in range(len(usersPosItem)):
                currentUser = usersPosItem[index2]
                if currentUser == sampled_user: #avoid learning from the same user sampled
                    continue
                how_many_items_liked_both = dense_batch_matrix[sampled_user, currentUser]      #quanti elementi piacciono sia all'utente campionato che a currentUser
                x_uij += (self.lambda_learning[currentUser] * how_many_items_liked_both)

            usersNegItem = self.getUsersSeenItem(batch_neg_items[index])    #e confronto anche con gli utenti a cui piace l'elemento che non piace all'user

            for index2 in range(len(usersNegItem)):
                currentUser = usersNegItem[index2]

                if currentUser == sampled_user:
                    print("Errore!!!, Non deve entrare qui")

                how_many_items_liked_both = dense_batch_matrix[sampled_user, currentUser]
                x_uij -= (self.lambda_learning[currentUser] * how_many_items_liked_both)

            gradient = (1 / (1 + exp(x_uij))) * (deriv_xuij) - (self.lambda_2*self.lambda_learning[sampled_user])     #calcolo il gradiente di quel campione
            self.update_model(gradient, sampled_user, usersPosItem)



    cdef double compute_pinv_cell(self, int row, int column):

        # SVD decomposition is U*s*V.t
        # Pseudoinverse is V*1/s*U.t

        cdef int latent_factor_index
        cdef double result = 0.0

        for latent_factor_index in range(self.SVD_latent_factors):

            result += self.SVD_Vh[latent_factor_index, row] / self.SVD_s[latent_factor_index] * self.SVD_U[column, latent_factor_index]

        return result



    cdef pseudoinverse_seq(self):
        cdef BPR_sample sample = self.sampleBatch_Cython()
        cdef double x_uij = 0.0, gradient

        cdef long i = sample.pos_item
        cdef long j = sample.neg_item
        cdef long sampled_user = sample.user

        cdef int [:] usersPosItem = self.getUsersSeenItem(i) #user a cui piace l'elemento
        cdef int [:] usersNegItem = self.getUsersSeenItem(j)

        #elementi che piacciono allo user campionato
        cdef int [:] currentUserLikeElement = self.URM_mask_indices[self.URM_mask_indptr[sample.user]:self.URM_mask_indptr[sample.user+1]]

        cdef double deriv_x_uij = 0.0
        cdef long index, current_item, index2, currentUser

        for index in range(len(currentUserLikeElement)): #per ogni elemento che piace all'user campionato
            current_item = currentUserLikeElement[index]

            if self.low_ram:
                deriv_x_uij += self.compute_pinv_cell(current_item, sampled_user)
            else:
                deriv_x_uij += self.pseudoInv[current_item, sampled_user]


            for index2 in range(len(usersPosItem)):
                #per ogni utente a cui piace l'elemento campionato (riga della colonna)

                currentUser = usersPosItem[index2]
                if currentUser == sample.user: # se questo utente è quello campionato lo salto
                    continue

                if self.low_ram:
                    x_uij+= self.compute_pinv_cell(current_item, currentUser) * self.lambda_learning[currentUser]
                else:
                    x_uij+= self.pseudoInv[current_item, currentUser] * self.lambda_learning[currentUser]

                #deriv_x_uij += self.pseudoInv[current_item, currentUser]

            for index2 in range(len(usersNegItem)):
                currentUser = usersNegItem[index2]
                if currentUser == sample.user:  #questa condizione non dovrebbe mai essere verificata
                    continue

                if self.low_ram:
                    x_uij-= self.compute_pinv_cell(current_item, currentUser) * self.lambda_learning[currentUser]
                else:
                    x_uij-= self.pseudoInv[current_item, currentUser] * self.lambda_learning[currentUser]



        gradient = (1 / (1 + exp(x_uij))) * (deriv_x_uij) - (self.lambda_2*self.lambda_learning[sample.user])

        if gradient != 0.0:
            self.update_model(gradient, sampled_user, usersPosItem)


    cdef update_model (self, double gradient, long sampled_user, int [:] usersPosItem):

        cdef double gradient_copy = gradient, cacheUpdate
        cdef int index2
        cdef long currentUser

        if self.useAdaGrad:
            cacheUpdate = gradient_copy ** 2
            self.sgd_cache[sampled_user] += cacheUpdate

            gradient = gradient_copy / (sqrt(self.sgd_cache[sampled_user]) + 1e-8)
            self.lambda_learning[sampled_user] += (self.learning_rate * gradient)   #applico il gradiente all'utente campionato

            if self.force_positive and self.lambda_learning[sampled_user] <0 : # forzo a 0 il lambda se negativo
                self.lambda_learning[sampled_user] = 0

            for index2 in range(len(usersPosItem)):     #applico il gradiente anche a coloro che hanno apprezzato i
                currentUser = usersPosItem[index2]
                if currentUser == sampled_user:
                    continue
                cacheUpdate = gradient_copy ** 2
                self.sgd_cache[currentUser] += cacheUpdate
                gradient = gradient_copy / (sqrt(self.sgd_cache[currentUser]) + 1e-8)
                self.lambda_learning[currentUser] += (self.learning_rate * gradient)

                if self.force_positive and self.lambda_learning[currentUser]<0:
                    self.lambda_learning[currentUser] = 0

        else:
            self.lambda_learning[sampled_user] += (self.learning_rate * gradient_copy)

            if self.force_positive and self.lambda_learning[sampled_user] <0 :
                self.lambda_learning[sampled_user] = 0

            for index2 in range(len(usersPosItem)):     #e confronto quanti item piacciono a coloro a cui piace quell'item
                currentUser = usersPosItem[index2]
                if currentUser == sampled_user:
                    continue
                self.lambda_learning[currentUser] += (self.learning_rate * gradient_copy)

                if self.force_positive and self.lambda_learning[currentUser]<0:
                    self.lambda_learning[currentUser] = 0


    cdef transpose_seq (self):

        cdef BPR_sample sample = self.sampleBatch_Cython()
        cdef double x_uij = 0.0
        cdef long user_index, user_id
        cdef long item_index, item_id #user_liked_current_element = 0, item_seen_by_sampled_user=0

        #cdef np.ndarray[double] lambdaArray = np.zeros(self.n_items)
        cdef int [:] usersPosItem, usersNegItem

        #cdef int [:] elements_liked_by_user, currentUserDislikeElements
        cdef double currentLambda, x_uij_deriv, gradient

        cdef array[double] template_zero = array('d')
        cdef array[double] lambda_local_sample = clone(template_zero, self.n_items, zero=True)


        usersPosItem = self.getUsersSeenItem(sample.pos_item)
        usersNegItem = self.getUsersSeenItem(sample.neg_item)

        for user_index in range(len(usersPosItem)):
            user_id = usersPosItem[user_index]
            if user_id == sample.user:
                continue

            currentLambda = self.lambda_learning[user_id] #lambda user corrente

            item_index = self.URM_mask_indptr[user_id]
            while item_index < self.URM_mask_indptr[user_id+1]:

                item_id = self.URM_mask_indices[item_index]
                item_index += 1
                lambda_local_sample[item_id] += currentLambda



        for user_index in range(len(usersNegItem)):
            user_id = usersNegItem[user_index]

            currentLambda = self.lambda_learning[user_id]

            item_index = self.URM_mask_indptr[user_id]
            while item_index < self.URM_mask_indptr[user_id+1]:

                item_id = self.URM_mask_indices[item_index]
                item_index += 1
                lambda_local_sample[item_id] += currentLambda




        x_uij = 0

        for item_index in range(len(self.seenItemsSampledUser)):
            item_id = self.seenItemsSampledUser[item_index]#in realta' è un item visto dall'utente campionato
            x_uij += lambda_local_sample[item_id]

        x_uij_deriv = len(self.seenItemsSampledUser)

        gradient = (1 / (1 + exp(x_uij))) * x_uij_deriv - self.lambda_2*self.lambda_learning[sample.user]

        if gradient != 0.0:
            self.update_model(gradient, sample.user, usersPosItem)

    #
    # cdef transpose_seq (self):
    #     # trasposta non in "batch". LENTA
    #         # non usata da novembre
    #     cdef BPR_sample sample = self.sampleBatch_Cython()
    #     cdef double x_uij = 0.0
    #     cdef long i = sample.pos_item
    #     cdef long j = sample.neg_item
    #     #cdef np.ndarray[double] lambdaArray = np.zeros(self.n_items)
    #     cdef int [:] usersPosItem = self.getUsersSeenItem(i)   #users a cui piace elemento positivo
    #     cdef long index = 0, index2, user_liked_current_element = 0, item_seen_by_sampled_user=0
    #     cdef int [:] elements_liked_by_user, usersNegItem, currentUserDislikeElements
    #     cdef double currentLambda = 0, x_uij_deriv = 0, gradient = 0
    #
    #     cdef array[double] template_zero = array('d')
    #     cdef array[double] lambda_local_sample = clone(template_zero, self.n_items, zero=True)
    #
    #
    #
    #     for index in range(len(usersPosItem)):
    #         user_liked_current_element = usersPosItem[index]
    #         if user_liked_current_element == sample.user:
    #             continue
    #
    #         # TUTTI gli elementi che piacciono all'i-esimo user
    #         elements_liked_by_user = self.URM_mask_indices[self.URM_mask_indptr[user_liked_current_element]:self.URM_mask_indptr[user_liked_current_element+1]]
    #         currentLambda = self.lambda_learning[user_liked_current_element] #lambda user corrente
    #
    #         for index2 in elements_liked_by_user:
    #             lambda_local_sample[index2] += currentLambda #copio lambda sugli elementi che sono piaciuti
    #
    #
    #     usersNegItem = self.getUsersSeenItem(j)
    #     for index in range(len(usersNegItem)):
    #         user_liked_current_element = usersNegItem[index]
    #         currentUserDislikeElements = (self.URM_mask_indices[self.URM_mask_indptr[user_liked_current_element]:self.URM_mask_indptr[user_liked_current_element+1]])
    #         currentLambda = self.lambda_learning[user_liked_current_element]
    #
    #         for index2 in elements_liked_by_user:
    #             lambda_local_sample[index2] -= currentLambda #copio lambda sugli elementi che sono piaciuti
    #
    #
    #     x_uij = 0
    #     for index in range(len(self.seenItemsSampledUser)):
    #         item_seen_by_sampled_user = self.seenItemsSampledUser[index]#in realta' è un item visto dall'utente campionato
    #         x_uij += lambda_local_sample[item_seen_by_sampled_user]
    #
    #     x_uij_deriv = len(self.seenItemsSampledUser)
    #     gradient = (1 / (1 + exp(x_uij))) * (x_uij_deriv) - (self.lambda_2*self.lambda_learning[sample.user])
    #     self.update_model(gradient, sample.user, usersPosItem)


    cpdef epochIteration_Cython(self):

        cdef long totalNumberOfBatch = int(self.n_users / self.batch_size)
        #print(self.batch_size, " ", self.numPositiveIteractions)
        cdef long start_time_epoch = time.time(), start_time_batch
        cdef int printStep = 100
        cdef double cacheUpdate
        cdef float gamma
        cdef long numCurrentBatch

        start_time_batch = time.time()

        if totalNumberOfBatch == 0:
            totalNumberOfBatch = 1
        #print("total number of batches ", totalNumberOfBatch, "for nnz: ", self.n_users)


        for numCurrentBatch in range(totalNumberOfBatch):

            if self.batch_size > 1 and self.enablePseudoInv == 0:         #impara in batch (con la trasposta!, non è implementato con pseudoinversa)
                self.transpose_batch()
            else:
                if self.batch_size > 1 and self.enablePseudoInv == 1:
                     raise ValueError("Not allowed pseudoInv in batch")
                if self.enablePseudoInv: #calcolo con pseudoinversa
                    self.pseudoinverse_seq()
                else:
                    self.transpose_seq()



            if (numCurrentBatch % printStep==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                itemPerSec = numCurrentBatch/(time.time()-start_time_epoch)
                printStep = int(itemPerSec*30)

                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size + 1)/totalNumberOfBatch,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()



    def get_lambda(self):

        return np.array(self.lambda_learning)


    def get_W_sparse(self, int TopK):
        """
        Returns W_sparse |item|x|item| computed via either the dense pseudoinverse or the
        SVD decomposition

        W_sparse = R+ * diag(lambda) * R_Train

        """

        cdef int itemIndex

        print("SLIM_Lambda_Cython: Computing W_sparse")

        # Use transpose
        if not self.enablePseudoInv:

           W_sparse = sps.diags(np.array(self.lambda_learning)).dot(self.URM_train)
           W_sparse = self.URM_train.T.dot(W_sparse)
           W_sparse.eliminate_zeros()

           return similarityMatrixTopK(W_sparse.T, k=TopK)



        # Use pseudoinverse
        if not self.low_ram:

            W_sparse = sps.diags(np.array(self.lambda_learning)).dot(np.array(self.pseudoInv).T)
            W_sparse = self.URM_train.T.dot(W_sparse)

            #W_sparse = np.array(self.pseudoInv).dot(sps.diags(np.array(self.lambda_learning)).dot(self.URM_train))

            return similarityMatrixTopK(W_sparse.T, k=TopK)



        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        cdef double[:] values = np.zeros((self.n_items*TopK))
        cdef int[:] rows = np.zeros((self.n_items*TopK,), dtype=np.int32)
        cdef int[:] cols = np.zeros((self.n_items*TopK,), dtype=np.int32)
        cdef long sparse_data_pointer = 0

        #
        # if not self.enablePseudoInv:
        #     URM_train_lambda = sps.diags(np.array(self.lambda_learning)).dot(self.URM_train)
        #
        # elif not self.low_ram:
        #     pseudoInv_lambda = sps.diags(np.array(self.lambda_learning)).dot(np.array(self.pseudoInv).T)
        #
        # else:
        SVD_s_inv = 1/np.array(self.SVD_s)




        self.URM_train = sps.csc_matrix(self.URM_train)


        for itemIndex in range(self.n_items):
            #
            # if not self.enablePseudoInv:
            #
            #     #this_item_weights = sps.diags(np.array(self.lambda_learning)).dot(self.URM_train)
            #     this_item_weights = self.URM_train[:,itemIndex].T.dot(URM_train_lambda).toarray().ravel()
            #
            # elif not self.low_ram:
            #
            #     #this_item_weights = sps.diags(np.array(self.lambda_learning)).dot(np.array(self.pseudoInv).T)
            #     this_item_weights = self.URM_train[:,itemIndex].T.dot(pseudoInv_lambda).ravel()
            #
            # else:
            pseudoinverse_row = np.array(np.multiply(self.SVD_Vh[:, itemIndex], SVD_s_inv)).ravel()
            pseudoinverse_row = pseudoinverse_row.dot(self.SVD_U.T)

            this_item_weights = sps.diags(np.array(self.lambda_learning)).dot(pseudoinverse_row)
            this_item_weights = self.URM_train.T.dot(this_item_weights)



            # Sort indices and select TopK
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # because we avoid sorting elements we already know we don't care about
            # - Partition the data to extract the set of TopK items, this set is unsorted
            # - Sort only the TopK items, discarding the rest
            # - Get the original item index

            this_item_weights = - np.array(this_item_weights)
            #
            # Get the unordered set of topK items
            top_k_partition = np.argpartition(this_item_weights, TopK-1)[0:TopK]
            # Sort only the elements in the partition
            top_k_partition_sorting = np.argsort(this_item_weights[top_k_partition])
            # Get original index
            top_k_idx = top_k_partition[top_k_partition_sorting]



            # Incrementally build sparse matrix
            for innerItemIndex in range(len(top_k_idx)):

                topKItemIndex = top_k_idx[innerItemIndex]

                values[sparse_data_pointer] = this_item_weights[topKItemIndex]
                rows[sparse_data_pointer] = itemIndex
                cols[sparse_data_pointer] = topKItemIndex

                sparse_data_pointer += 1


        values = np.array(values[0:sparse_data_pointer])
        rows = np.array(rows[0:sparse_data_pointer])
        cols = np.array(cols[0:sparse_data_pointer])

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                shape=(self.n_items, self.n_items),
                                dtype=np.float32)

        return W_sparse



    cdef BPR_sample sampleBatch_Cython(self):

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected
        cdef double RAND_MAX_DOUBLE = RAND_MAX
        index = int(rand() / RAND_MAX_DOUBLE * self.numEligibleUsers )
        sample.user = self.eligibleUsers[index]
        self.seenItemsSampledUser = self.getSeenItemsOfUser(sample.user)
        self.numSeenItemsSampledUser = len(self.seenItemsSampledUser)
        index = int(rand() / RAND_MAX_DOUBLE * self.numSeenItemsSampledUser )
        sample.pos_item = self.seenItemsSampledUser[index]
        negItemSelected = False
        while (not negItemSelected):
            sample.neg_item = int(rand() / RAND_MAX_DOUBLE  * self.n_items )
            index = 0
            while index < self.numSeenItemsSampledUser and self.seenItemsSampledUser[index]!=sample.neg_item:
                index+=1
            if index == self.numSeenItemsSampledUser:
                negItemSelected = True
        return sample


#---funzioni---
##################################################################################################################
#####################
#####################            SPARSE MATRIX
#####################
##################################################################################################################


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


import scipy.sparse as sps

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


# Node struct
ctypedef struct matrix_element_tree_s:
    long column
    double data
    matrix_element_tree_s *higher
    matrix_element_tree_s *lower

ctypedef struct head_pointer_tree_s:
    matrix_element_tree_s *head


# Function to allocate a new node
cdef matrix_element_tree_s * pointer_new_matrix_element_tree_s(long column, double data, matrix_element_tree_s *higher,  matrix_element_tree_s *lower):

    cdef matrix_element_tree_s * new_element

    new_element = < matrix_element_tree_s * > malloc(sizeof(matrix_element_tree_s))
    new_element.column = column
    new_element.data = data
    new_element.higher = higher
    new_element.lower = lower

    return new_element


# Functions to compare structs to be used in C qsort
cdef int compare_struct_on_column(const void *a_input, const void *b_input):
    """
    The function compares the column contained in the two struct passed.
    If a.column > b.column returns >0
    If a.column < b.column returns <0

    :return int: a.column - b.column
    """

    cdef head_pointer_tree_s *a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s *b_casted = <head_pointer_tree_s *> b_input

    return a_casted.head.column  - b_casted.head.column



cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0
    If a.data < b.data returns <0

    :return int: +1 or -1
    """

    cdef head_pointer_tree_s * a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s * b_casted = <head_pointer_tree_s *> b_input

    if (a_casted.head.data - b_casted.head.data) > 0.0:
        return +1
    else:
        return -1



#################################
#################################       CLASS DECLARATION
#################################

cdef class Sparse_Matrix_Tree_CSR:

    cdef long num_rows, num_cols

    # Array containing the struct (object, not pointer) corresponding to the root of the tree
    cdef head_pointer_tree_s* row_pointer

    def __init__(self, long num_rows, long num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_pointer = < head_pointer_tree_s *> malloc(self.num_rows * sizeof(head_pointer_tree_s))

        # Initialize all rows to empty
        for index in range(self.num_rows):
            self.row_pointer[index].head = NULL


    cpdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.

        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element
        cdef matrix_element_tree_s* old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data += value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:

            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:

            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'




    cpdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.

        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """


        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError(
                "Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                    self.num_rows, self.num_cols, row, col))


        cdef matrix_element_tree_s* current_element
        cdef int stopSearch = False

        # If the row is empty, return default
        if self.row_pointer[row].head == NULL:
            return 0.0


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True


        # If the cell exist, return its value
        if current_element.column == col:
            return current_element.data

        # The cell is not found, return default
        else:
            return 0.0




    cpdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object

        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row

        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # row contains data
            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)


                # Flatten the tree data
                subtree_column, subtree_data = self.from_linked_list_to_python_list(self.row_pointer[row].head)
                data.extend(subtree_data)
                indices.extend(subtree_column)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



    cpdef rebalance_tree(self, long TopK = False):
        """
        The function builds a balanced binary tree from the current one, for all matrix rows

        :param TopK: either False or an integer number. Number of the highest elements to preserve
        """

        cdef long row

        #start_time = time.time()

        for row in range(self.num_rows):

            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


    cdef matrix_element_tree_s * subtree_to_list_flat(self, matrix_element_tree_s * root):
        """
        The function flatten the structure of the subtree whose root is passed as a paramether
        The list is bidirectional and ordered with respect to the column
        The column ordering follows from the insertion policy

        :param root: tree root
        :return list, list: data and corresponding column. Empty list if root is None
        """

        if root == NULL:
            return NULL

        cdef matrix_element_tree_s *flat_list_head
        cdef matrix_element_tree_s *current_element

        # Flatten lower subtree
        flat_list_head = self.subtree_to_list_flat(root.lower)

        # If no lower elements exist, the head is the current element
        if flat_list_head == NULL:
            flat_list_head = root
            root.lower = NULL

        # Else move to the tail and add the subtree root
        else:
            current_element = flat_list_head
            while current_element.higher != NULL:
                current_element = current_element.higher

            # Attach the element with the bidirectional pointers
            current_element.higher = root
            root.lower = current_element

        # Flatten higher subtree and attach it to the tail of the flat list
        root.higher = self.subtree_to_list_flat(root.higher)

        # Attach the element with the bidirectional pointers
        if root.higher != NULL:
            root.higher.lower = root

        return flat_list_head



    cdef from_linked_list_to_python_list(self, matrix_element_tree_s * head):

        data = []
        column = []

        while head != NULL:
            data.append(head.data)
            column.append(head.column)

            head = head.higher

        return column, data



    cdef subtree_free_memory(self, matrix_element_tree_s* root):
        """
        The function frees all struct in the subtree whose root is passed as a parameter, root included

        :param root: tree root
        """

        if root != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(root.higher)
            self.subtree_free_memory(root.lower)

            # Once the lower elements have been reached, start freeing from the bottom
            free(root)



    cdef list_free_memory(self, matrix_element_tree_s * head):
        """
        The function frees all struct in the list whose head is passed as a parameter, head included

        :param head: list head
        """

        if head != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(head.higher)

            # Once the tail element have been reached, start freeing from them
            free(head)



    cdef matrix_element_tree_s* build_tree_from_list_flat(self, matrix_element_tree_s* flat_list_head):
        """
        The function builds a tree containing the passed data. This is the recursive function, the
        data should be sorted by te caller
        To ensure the tree is balanced, data is sorted according to the column

        :param row: row in which to create new tree
        :param column_vector: column coordinates
        :param data_vector: cell data
        """

        if flat_list_head == NULL:
            return NULL


        cdef long list_length = 0
        cdef long middle_element_step = 0

        cdef matrix_element_tree_s *current_element
        cdef matrix_element_tree_s *middleElement
        cdef matrix_element_tree_s *tree_root

        current_element = flat_list_head
        middleElement = flat_list_head

        # Explore the flat list moving the middle elment every tho jumps
        while current_element != NULL:
            current_element = current_element.higher
            list_length += 1
            middle_element_step += 1

            if middle_element_step == 2:
                middleElement = middleElement.higher
                middle_element_step = 0

        tree_root = middleElement

        # To execute the recursion it is necessary to cut the flat list
        # The last of the lower elements will have to be a tail
        if middleElement.lower != NULL:
            middleElement.lower.higher = NULL

            tree_root.lower = self.build_tree_from_list_flat(flat_list_head)


        # The first of the higher elements will have to be a head
        if middleElement.higher != NULL:
            middleElement.higher.lower = NULL

            tree_root.higher = self.build_tree_from_list_flat(middleElement.higher)


        return tree_root




    cdef matrix_element_tree_s* topK_selection_from_list(self, matrix_element_tree_s* head, long TopK):
        """
        The function selects the topK highest elements in the given list

        :param head: head of the list
        :param TopK: number of highest elements to preserve
        :return matrix_element_tree_s*: head of the new list
        """

        cdef head_pointer_tree_s *vector_pointer_to_list_elements
        cdef matrix_element_tree_s *current_element
        cdef long list_length, index, selected_count

        # Get list size
        current_element = head
        list_length = 0

        while current_element != NULL:
            list_length += 1
            current_element = current_element.higher


        # If list elements are not enough to perform a selection, return
        if list_length < TopK:
            return head

        # Allocate vector that will be used for sorting
        vector_pointer_to_list_elements = < head_pointer_tree_s *> malloc(list_length * sizeof(head_pointer_tree_s))

        # Fill vector wit pointers to list elements
        current_element = head
        for index in range(list_length):
            vector_pointer_to_list_elements[index].head = current_element
            current_element = current_element.higher


        # Sort array elements on their data field
        qsort(vector_pointer_to_list_elements, list_length, sizeof(head_pointer_tree_s), compare_struct_on_data)

        # Sort only the TopK according to their column field
        # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
        qsort(&vector_pointer_to_list_elements[list_length-TopK], TopK, sizeof(head_pointer_tree_s), compare_struct_on_column)


        # Rebuild list attaching the consecutive elements
        index = list_length-TopK

        # Detach last TopK element from previous ones
        vector_pointer_to_list_elements[index].head.lower = NULL

        while index<list_length-1:
            # Rearrange bidirectional pointers
            vector_pointer_to_list_elements[index+1].head.lower = vector_pointer_to_list_elements[index].head
            vector_pointer_to_list_elements[index].head.higher = vector_pointer_to_list_elements[index+1].head

            index += 1

        # Last element in vector will be the hew head
        vector_pointer_to_list_elements[list_length - 1].head.higher = NULL

        # Get hew list head
        current_element = vector_pointer_to_list_elements[list_length-TopK].head

        # If there are exactly enough elements to reach TopK, index == 0 will be the tail
        # Else, index will be the tail and the other elements will be removed
        index = list_length - TopK - 1
        if index > 0:

            index -= 1
            while index >= 0:
                free(vector_pointer_to_list_elements[index].head)
                index -= 1

        # Free array
        free(vector_pointer_to_list_elements)


        return current_element






##################################################################################################################
#####################
#####################            TEST FUNCTIONS
#####################
##################################################################################################################


    cpdef test_list_tee_conversion(self, long row):
        """
        The function tests the inner data structure conversion from tree to C linked list and back to tree

        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head
        cdef matrix_element_tree_s *tree_root
        cdef matrix_element_tree_s *current_element
        cdef matrix_element_tree_s *previous_element

        head = self.subtree_to_list_flat(self.row_pointer[row].head)
        current_element = head

        cdef numElements_higher = 0
        cdef numElements_lower = 0

        while current_element != NULL:
            numElements_higher += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower += 1
            current_element = current_element.lower

        assert numElements_higher == numElements_lower, 'Bidirectional linked list not consistent.' \
                                                        ' From head to tail element count is {}, from tail to head is {}'.format(
                                                        numElements_higher, numElements_lower)

        print("Bidirectional list link - Passed")

        column_original, data_original = self.from_linked_list_to_python_list(head)

        assert numElements_higher == len(column_original), \
            'Data structure size inconsistent. LinkedList is {}, Python list is {}'.format(numElements_higher, len(column_original))

        for index in range(len(column_original)-1):
            assert column_original[index] < column_original[index+1],\
                'Columns not ordered correctly. Tree not flattened properly'

        print("Bidirectional list ordering - Passed")

        # Transform list into tree and back into list, as it is easy to test
        tree_root = self.build_tree_from_list_flat(head)
        head = self.subtree_to_list_flat(tree_root)

        cdef numElements_higher_after = 0
        cdef numElements_lower_after = 0

        current_element = head

        while current_element != NULL:
            numElements_higher_after += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower_after += 1
            current_element = current_element.lower

        print("Bidirectional list from tree link - Passed")

        assert numElements_higher_after == numElements_lower_after, \
            'Bidirectional linked list after tree construction not consistent. ' \
            'From head to tail element count is {}, from tail to head is {}'.format(
            numElements_higher_after, numElements_lower_after)

        assert numElements_higher == numElements_higher_after, \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                numElements_higher, numElements_higher_after)

        column_after_tree, data_after_tree = self.from_linked_list_to_python_list(head)

        assert len(column_original) == len(column_after_tree), \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                len(column_original), len(column_after_tree))

        for index in range(len(column_original)):
            assert column_original[index] == column_after_tree[index],\
                'After tree construction columns are not ordered properly'
            assert data_original[index] == data_after_tree[index],\
                'After tree construction data content is changed'

        print("Bidirectional list from tree ordering - Passed")



    cpdef test_topK_from_list_selection(self, long row, long topK):
        """
        The function tests the topK selection from list

        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head

        head = self.subtree_to_list_flat(self.row_pointer[row].head)

        column_original, data_original = self.from_linked_list_to_python_list(head)

        head = self.topK_selection_from_list(head, topK)

        column_topK, data_topK = self.from_linked_list_to_python_list(head)

        assert len(column_topK) == len(data_topK),\
            "TopK data and column lists have different length. Columns length is {}, data is {}".format(len(column_topK), len(data_topK))
        assert len(column_topK) <= topK,\
            "TopK extracted list is longer than desired value. Desired is {}, while list is {}".format(topK, len(column_topK))

        print("TopK extracted length - Passed")

        # Sort with respect to the content to select topK
        idx_sorted = np.argsort(data_original)
        idx_sorted = np.flip(idx_sorted, axis=0)
        top_k_idx = idx_sorted[0:topK]

        column_topK_numpy = np.array(column_original)[top_k_idx]
        data_topK_numpy = np.array(data_original)[top_k_idx]

        # Sort with respect to the column to ensure it is ordered as the tree flattened list
        idx_sorted = np.argsort(column_topK_numpy)
        column_topK_numpy = column_topK_numpy[idx_sorted]
        data_topK_numpy = data_topK_numpy[idx_sorted]


        assert len(column_topK_numpy) <= len(column_topK),\
            "TopK extracted list and numpy one have different length. Extracted list lenght is {}, while numpy is {}".format(
                len(column_topK_numpy), len(column_topK))


        for index in range(len(column_topK)):

            assert column_topK[index] == column_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as column value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, column_topK[index], column_topK_numpy[index])

            assert data_topK[index] == data_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as data value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, data_topK[index], data_topK_numpy[index])

        print("TopK extracted content - Passed")




