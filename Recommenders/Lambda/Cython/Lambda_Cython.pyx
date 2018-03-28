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
    cdef int sparse_weights
    cdef long[:] eligibleUsers
    cdef long numEligibleUsers
    cdef int batch_size

    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser
    cdef double alpha #coefficiente di regolarizzazione

    cdef int[:] URM_mask_indices, URM_mask_indptr
    cdef int[:] URM_mask_transp_indices, URM_mask_transp_indptr #faster access to rows index with 1 of certain item (col)
    cdef double[:] lambda_learning # lambda learned
    cdef double[:,:] pseudoInv #pseudoinverse
    cdef int enablePseudoInv

    cdef double [:] sgd_cache #cache adagrad

    cdef S_sparse

    def __init__(self, URM_mask, sparse_weights, eligibleUsers,
                 learning_rate = 0.05, alpha=0.0002,
                 batch_size = 1, sgd_mode='sgd', enablePseudoInv = False, pseudoInv = None, initialize="zero"):

        super(Lambda_BPR_Cython_Epoch, self).__init__()
        URM_mask = check_matrix(URM_mask, 'csr')
        self.numPositiveIteractions = int(URM_mask.nnz)
        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.alpha=alpha
        self.sparse_weights = sparse_weights
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
            self.lambda_learning = np.random.rand(self.n_users)
        else:
            raise ValueError(
                "init not valid. Values: zero, one, random")

        self.enablePseudoInv = 0
        if enablePseudoInv:
            self.pseudoInv = pseudoInv
            self.enablePseudoInv = 1

        if self.sparse_weights:
            self.S_sparse = Sparse_Matrix_Tree_CSR(self.n_users, self.n_users)

        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            print("adagrad enabled")
        elif sgd_mode=='rmsprop':
            self.rmsprop = True #not implemented
        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD not valid.")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eligibleUsers = eligibleUsers
        self.numEligibleUsers = len(eligibleUsers)

        if self.useAdaGrad:
            self.sgd_cache = np.zeros((self.n_users), dtype=float)

        elif self.rmsprop:
            self.sgd_cache = np.zeros((self.n_users), dtype=float)
            self.gamma = 0.90


    cdef int[:] getSeenItemsOfUser(self, long index): #given an user, get the items he likes
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]

    cdef int[:] getUsersSeenItem(self, long index): # given an item, get the users who like that item
        return self.URM_mask_transp_indices[self.URM_mask_transp_indptr[index]:self.URM_mask_transp_indptr[index + 1]]

    def epochIteration_Cython(self):

        cdef long totalNumberOfBatch = int(self.numPositiveIteractions / self.batch_size)
        print(self.batch_size, " ", self.numPositiveIteractions)
        cdef long start_time_epoch = time.time()

        cdef BPR_sample sample
        cdef long i=0, j=0, user=0, currentUser=0
        cdef long index=0, numCurrentBatch=0, current_item = 0
        cdef double gradient=0, xi=0, xj=0, x_uij=0, gradientBk = 0
        cdef double second_part_gradient=0
        cdef int [:] usersPosItem, usersNegItem, currentUserDislikeElements, currentUserLikeElement

        cdef np.ndarray[double] lambdaArray
        cdef np.ndarray[long] countArray
        cdef double currentLambda=0, avgLambda=0
        cdef int printStep = 500000

        cdef long[:] batch_users, batch_pos_items, batch_neg_items, user_arr
        cdef int[:] tmp_batch, tmp2_batch
        cdef long[:,:] dense_batch_matrix # matrix multiplication result
        cdef long index2=0, sampled_user=0
        cdef batch_matrix
        cdef map [long, long] all_users #C++ map
        cdef map [long, long] itemsChoosenSampling # C++ map
        cdef map[long,long].iterator itr
        cdef double update1=0, update2=0
        cdef long how_many_times=0

        cdef double cacheUpdate
        cdef float gamma

        if self.rmsprop:
            gamma = 0.90

        start_time_batch = time.time()

        if totalNumberOfBatch == 0:
            totalNumberOfBatch = 1

        print("total number of batches ", totalNumberOfBatch, "for nnz: ", self.numPositiveIteractions)

        for numCurrentBatch in range(totalNumberOfBatch):
            gradient = 0
            second_part_gradient = 0
            t = time.time()

            batch_users = np.zeros(self.batch_size, dtype=int)
            if self.batch_size > 1 and self.enablePseudoInv == 0:         #impara in batch (con la trasposta!, non è implementato con pseudoinversa)
                batch_pos_items = np.zeros(self.batch_size, dtype=int)
                batch_neg_items = np.zeros(self.batch_size, dtype=int)
                all_users.clear()
                itemsChoosenSampling.clear()
                #esegui tutti i campionamenti e tieni traccia degli utenti che devono essere inseriti nella matrice (utenti coinvolti)
                for index in range(self.batch_size):
                    # campiona e salva i campionamenti
                    sample = self.sampleBatch_Cython()
                    batch_users[index] = sample.user            #questi 3 array contengono user, elementi pos e elementi neg. campionati
                    batch_pos_items[index] = sample.pos_item
                    batch_neg_items[index] = sample.neg_item

                    all_users[sample.user] = sample.user        #questa mappa contiene tutti gli user che sono richiesti per l'elaborazione futura

                    if itemsChoosenSampling.count(sample.pos_item) == 0: # cerco se nella mappa degli item campionati è gia' stato campionato lo stesso elemento
                                                                        # (se sì non ha senso re-inserire gli utenti a cui piace l'item nell'altra mappa visto che sono stati inseriti in precedenza)
                        itemsChoosenSampling[sample.pos_item] = sample.pos_item

                        tmp_batch = self.getUsersSeenItem(sample.pos_item)
                        for index2 in range(len(tmp_batch)):            #quindi anche quelli che hanno visto l'elemento positivo
                            all_users[tmp_batch[index2]] = tmp_batch[index2]

                    if itemsChoosenSampling.count(sample.neg_item) == 0: # uguale per il campione "negativo"
                        itemsChoosenSampling[sample.neg_item] = sample.neg_item

                        tmp_batch = self.getUsersSeenItem(sample.neg_item)
                        for index2 in range(len(tmp_batch)):            #e quelli che hanno visto l'elemento negativo
                            all_users[tmp_batch[index2]] = tmp_batch[index2]

                #crea la matrice sparsa
                batch_matrix = Sparse_Matrix_Tree_CSR(self.n_users, self.n_items)
                index = 0

                print("Step 2")
                #inserisco i profili nella matrice
                itr=all_users.begin()
                while itr!=all_users.end():         #scorro gli user della mappa e copio gli elementi che piacciono ad ogni user nella matrice
                    currentUser = deref(itr).first #un utente
                    tmp2_batch = self.getSeenItemsOfUser(currentUser) #elementi che piacciono all'utente
                    for index2 in range(len(tmp2_batch)):   #copia quello che piace all'user nella matrice
                        batch_matrix.add_value(currentUser, tmp2_batch[index2], 1)
                    inc(itr)

                #calcolo il prodotto con la trasposta

                print("Step 3")

                scipy_tmp = batch_matrix.get_scipy_csr()
                scipy_tmp = scipy_tmp.astype(np.long)

                scipy_tmp = scipy_tmp.dot(scipy_tmp.T)
                print("done")
                dense_batch_matrix = scipy_tmp.todense() # rendo densa la matrice per andare più veloce negli accessi. POTREBBE CREARE PROBLEMI IN RAM con grandi campionamenti su grandi matrici
                print("densifyed")
                del(scipy_tmp)

                print("iterating... ",len(batch_users))
                for index in range(len(batch_users)):#per ogni campione calcolo e applico il gradiente...
                    if index%50000 == 0:
                        print("computing: ", index ," of ", len(batch_users))
                    update1 = 0
                    sampled_user = batch_users[index]       #prendo l'utente del campione
                    update2 = len(self.getSeenItemsOfUser(sampled_user))


                    usersPosItem = self.getUsersSeenItem(batch_pos_items[index]) #items che piacciono all'utente campionato in precedenza
                    for index2 in range(len(usersPosItem)):     #e confronto quanti item piacciono a coloro a cui piace quell'item
                        currentUser = usersPosItem[index2]

                        if currentUser == sampled_user: #avoid learning from the same user sampled
                            continue

                        how_many_times = dense_batch_matrix[sampled_user, currentUser]      #quanti elementi piacciono sia all'utente campionato che a currentUser
                        update1 += (self.lambda_learning[currentUser] * how_many_times)
                        #update2 += how_many_times

                    usersNegItem = self.getUsersSeenItem(batch_neg_items[index])    #e confronto anche con gli utenti a cui piace l'elemento che non piace all'user

                    for index2 in range(len(usersNegItem)):
                        #currentUser = all_users[usersNegItem[index2]]
                        currentUser = usersNegItem[index2]

                        if currentUser == sampled_user:
                            print("Errore!!!, Non deve entrare qui")

                        how_many_times = dense_batch_matrix[sampled_user, currentUser]
                        update1 -= (self.lambda_learning[currentUser] * how_many_times)
                        #update2 -= how_many_times

                    gradient = (1 / (1 + exp(update1))) * (update2) - (self.alpha*self.lambda_learning[sampled_user])     #calcolo il gradiente di quel campione
                    gradientBk = gradient
                    #avgLambda+=gradient

                    if self.useAdaGrad:
                        cacheUpdate = gradientBk ** 2
                        self.sgd_cache[sampled_user] += cacheUpdate

                        gradient = gradientBk / (sqrt(self.sgd_cache[sampled_user]) + 1e-8)
                        self.lambda_learning[sampled_user] += (self.learning_rate * gradient)   #applico il gradiente all'utente campionato

                        if self.lambda_learning[sampled_user] <0 : # forzo a 0 il lambda se negativo
                            self.lambda_learning[sampled_user] = 0
                        for index2 in range(len(usersPosItem)):     #applico il gradiente anche a coloro che hanno apprezzato i
                            currentUser = usersPosItem[index2]
                            if currentUser == sampled_user:
                                continue
                            #self.sgd_cache[currentUser] += cacheUpdate
                            #gradient = gradientBk / (sqrt(self.sgd_cache[currentUser]) + 1e-8)

                            cacheUpdate = gradientBk ** 2
                            self.sgd_cache[currentUser] += cacheUpdate
                            gradient = gradientBk / (sqrt(self.sgd_cache[currentUser]) + 1e-8)
                            self.lambda_learning[currentUser] += (self.learning_rate * gradient)
                            if self.lambda_learning[currentUser]<0:
                                self.lambda_learning[currentUser] = 0

                    else:
                        self.lambda_learning[sampled_user] += (self.learning_rate * gradientBk)
                        if self.lambda_learning[sampled_user] <0 :
                            self.lambda_learning[sampled_user] = 0
                        for index2 in range(len(usersPosItem)):     #e confronto quanti item piacciono a coloro a cui piace quell'item
                            currentUser = usersPosItem[index2]
                            if currentUser == sampled_user:
                                continue
                            self.lambda_learning[currentUser] += (self.learning_rate * gradientBk)
                            if self.lambda_learning[currentUser]<0:
                                self.lambda_learning[currentUser] = 0

            else:
                if self.batch_size > 1 and self.enablePseudoInv == 1:
                     raise ValueError("Not allowed pseudoInv in batch")

                if self.enablePseudoInv: #calcolo con pseudoinversa
                    sample = self.sampleBatch_Cython()
                    x_uij = 0.0

                    i = sample.pos_item
                    j = sample.neg_item
                    sampled_user = sample.user

                    usersPosItem = self.getUsersSeenItem(i) #user a cui piace l'elemento
                    usersNegItem = self.getUsersSeenItem(j)
                    currentUserLikeElement = (self.URM_mask_indices[self.URM_mask_indptr[sample.user]:self.URM_mask_indptr[sample.user+1]]) #elementi che piacciono allo user campionato

                    second_part_gradient = 0.0

                    for index in range(len(currentUserLikeElement)): #per ogni elemento che piace all'user campionato
                        current_item = currentUserLikeElement[index]
                        second_part_gradient += self.pseudoInv[current_item, sampled_user]

                        for index2 in range(len(usersPosItem)): #per ogni utente a cui piace l'elemento campionato (riga della colonna)
                            currentUser = usersPosItem[index2]
                            if currentUser == sample.user: # se questo utente è quello campionato lo salto
                                continue
                            x_uij+= (self.pseudoInv[current_item, currentUser] * self.lambda_learning[currentUser])
                            #second_part_gradient += self.pseudoInv[current_item, currentUser]


                        for index2 in range(len(usersNegItem)):
                            currentUser = usersNegItem[index2]
                            if currentUser == sample.user:  #questa condizione non dovrebbe mai essere verificata
                                continue
                            x_uij-= (self.pseudoInv[current_item, currentUser] * self.lambda_learning[currentUser])
                            #second_part_gradient -= self.pseudoInv[current_item, currentUser]

                    gradient = (1 / (1 + exp(x_uij))) * (second_part_gradient) - (self.alpha*self.lambda_learning[sample.user])
                    gradientBk = gradient

                    if self.useAdaGrad:
                        cacheUpdate = gradientBk ** 2
                        self.sgd_cache[sampled_user] += cacheUpdate
                        gradient = gradientBk / (sqrt(self.sgd_cache[sampled_user]) + 1e-8)

                        self.lambda_learning[sampled_user] += (self.learning_rate * gradient)

                        if self.lambda_learning[sampled_user] <0 :
                            self.lambda_learning[sampled_user] = 0

                        for index2 in range(len(usersPosItem)):
                            currentUser = usersPosItem[index2]
                            if currentUser == sampled_user:
                                continue

                            cacheUpdate = gradientBk ** 2
                            self.sgd_cache[currentUser] += cacheUpdate
                            gradient = gradientBk / (sqrt(self.sgd_cache[currentUser]) + 1e-8)

                            self.lambda_learning[currentUser] += (self.learning_rate * gradient)
                            if self.lambda_learning[currentUser]<0:
                                self.lambda_learning[currentUser] = 0
                    else:
                        self.lambda_learning[sampled_user] += (self.learning_rate * gradient)
                        if self.lambda_learning[sampled_user] <0 :
                            self.lambda_learning[sampled_user] = 0

                        for index2 in range(len(usersPosItem)):     #e confronto quanti item piacciono a coloro a cui piace quell'item
                            currentUser = usersPosItem[index2]
                            if currentUser == sampled_user:
                                continue
                            self.lambda_learning[currentUser] += (self.learning_rate * gradientBk)
                            if self.lambda_learning[currentUser]<0:
                                self.lambda_learning[currentUser] = 0


                else:
                    # trasposta non in batch. LENTA
                    # non usata da novembre
                    sample = self.sampleBatch_Cython()
                    x_uij = 0.0

                    i = sample.pos_item
                    j = sample.neg_item

                    lambdaArray = np.zeros(self.n_items)
                    countArray = np.zeros(self.n_items, dtype=np.int64)

                    usersPosItem = self.getUsersSeenItem(i)   #users a cui piace elemento positivo

                    for index in range(len(usersPosItem)):
                        currentUser = usersPosItem[index]

                        if currentUser == sample.user:
                            continue

                        currentUserLikeElement = (self.URM_mask_indices[self.URM_mask_indptr[currentUser]:self.URM_mask_indptr[currentUser+1]]) # TUTTI gli elementi che piacciono all'i-esimo user
                        currentLambda = self.lambda_learning[currentUser] #lambda user corrente
                        lambdaArray[currentUserLikeElement] += currentLambda #copio lambda sugli elementi che sono piaciuti
                        countArray[currentUserLikeElement] += 1

                    usersNegItem = self.getUsersSeenItem(j)

                    for index in range(len(usersNegItem)):
                        currentUser = usersNegItem[index]

                        currentUserDislikeElements = (self.URM_mask_indices[self.URM_mask_indptr[currentUser]:self.URM_mask_indptr[currentUser+1]])
                        currentLambda = self.lambda_learning[currentUser]
                        lambdaArray[currentUserDislikeElements] -= currentLambda
                        countArray[currentUserDislikeElements] -= 1

                    second_part_gradient = 0
                    x_uij = 0

                    for index in range(len(self.seenItemsSampledUser)):
                        currentUser = self.seenItemsSampledUser[index]#in realta' è un item visto dall'utente campionato
                        x_uij += lambdaArray[currentUser]

                    second_part_gradient = len(self.seenItemsSampledUser)

                    gradient = (1 / (1 + exp(x_uij))) * (second_part_gradient) - (self.alpha*self.lambda_learning[sample.user])

                    if self.useAdaGrad:
                        cacheUpdate = gradient ** 2
                        self.sgd_cache[i] += cacheUpdate
                        self.sgd_cache[j] += cacheUpdate
                        gradient = gradient / (sqrt(self.sgd_cache[i]) + 1e-8)

                    elif self.rmsprop:
                        cacheUpdate = self.sgd_cache[i] * gamma + (1 - gamma) * gradient ** 2
                        self.sgd_cache[i] = cacheUpdate
                        self.sgd_cache[j] = cacheUpdate
                        gradient = gradient / (sqrt(self.sgd_cache[i]) + 1e-8)

                    self.lambda_learning[sample.user]+=self.learning_rate * gradient

                    if self.lambda_learning[sample.user] <0 :
                        self.lambda_learning[sample.user] = 0


                    usersPosItem = self.getUsersSeenItem(i)   #users a cui piace elemento positivo
                    for index in range(len(usersPosItem)):
                        currentUser = usersPosItem[index]
                        if currentUser == sample.user:
                            continue
                        self.lambda_learning[currentUser] += (self.learning_rate * gradient)
                        if self.lambda_learning[currentUser] <0:
                            self.lambda_learning[currentUser] = 0

            if numCurrentBatch % 4500 == 0:
                print(numCurrentBatch, " of ", totalNumberOfBatch)

            if((numCurrentBatch%printStep==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        print("Return S matrix to python caller")


        self.S_sparse = Sparse_Matrix_Tree_CSR(self.n_users, self.n_users)
        for index in range(self.n_users):
            self.S_sparse.add_value(index, index, self.lambda_learning[index])

        print("returning...")
        return self.S_sparse.get_scipy_csr()



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




