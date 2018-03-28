import numpy as np
import scipy.sparse as sps
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool



def load_dataframe(path, sep, drop_columns=[], implicit_field=None, implicit_treshold=3, names=None,
                    header='infer'):
    df = pd.read_csv(path, sep=sep, header=header, names=names)
    if len(drop_columns) > 0:
        df.drop(drop_columns, axis=1, inplace=True)
    if implicit_field is not None:
        df.loc[df[implicit_field] <= implicit_treshold, [implicit_field]] = 0
        df.loc[df[implicit_field] > implicit_treshold, [implicit_field]] = 1

        #df[implicit_field] = df[implicit_field].apply(lambda x: 0 if x < implicit_treshold else 1)
        df = df[df[implicit_field] > 0]
    return df



cores = cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def work_parallel (dataframe):
    print("cleaning ", field)
    dataframe[field] = dataframe[field].apply(lambda x: list_id.index(x))
    print("end")
    return dataframe

def clean_dataframe(dataframe, fields=[]):
    global list_id
    global field
    for field in fields:
        list_id = sorted(set(dataframe[field]))
        #l = len(list_id)
        #for i in range(0, l):
        #    dataframe.loc[dataframe[field] == list_id[i], [field]] = i
        #    if i % 1000 == 0:
        #        print(i)
        dataframe = parallelize(dataframe, work_parallel)
    return dataframe


def save_dataframe(dataframe, path, sep):
    dataframe.to_csv(path, index=False, sep=sep)


class Dataset():
    def __init__(self, dataframe, user_key='userId', item_key='movieId', rating_key='rating'):
        self.URM = self.df_to_csr(dataframe, user_key=user_key, item_key=item_key, rating_key=rating_key)
        #shuffle rows
        #index = np.arange(np.shape(self.URM)[0])
        #np.random.shuffle(index)
        #self.URM = self.URM[index, :]

    def k_fold_cv(self, k=10):
        # First: partition the matrix in k fold
        n_users = self.URM.shape[0]
        fold_size = (n_users // k)

        list_users = list(range(0, n_users))

        for fold in range(0, k):
            print(list_users[:fold * fold_size] + list_users[(fold + 1) * fold_size:])

            train_matrix = self.URM[list_users[:fold * fold_size] + list_users[(fold + 1) * fold_size:], :]
            print(train_matrix.shape)
            test_matrix = self.URM[list_users[fold * fold_size:(fold + 1) * fold_size], :]

            # split test into profiles and test

            train_test_split = 0.80
            num_interactions = test_matrix.nnz
            mask = np.random.choice([True, False], num_interactions, [train_test_split, 1 - train_test_split])
            test_matrix = test_matrix.tocoo()
            print(test_matrix.row)
            print(test_matrix.row[mask])
            test_profiles = sps.coo_matrix((test_matrix.data[mask], (test_matrix.row[mask], test_matrix.col[mask])), shape=(test_matrix.shape[0], test_matrix.shape[1]))
            test_profiles = test_profiles.tocsr()
            mask = np.logical_not(mask)
            test_ratings = sps.coo_matrix((test_matrix.data[mask], (test_matrix.row[mask], test_matrix.col[mask])), shape=(test_matrix.shape[0], test_matrix.shape[1]))
            test_ratings = test_ratings.tocsr()

            yield train_matrix, test_profiles, test_ratings

    def df_to_csr(self, df, user_key='userId', item_key='movieId', rating_key='rating'):
        """
        Convert a pandas DataFrame to a scipy.sparse.csr_matrix
        """
        rows = df[user_key].values
        columns = df[item_key].values
        ratings = df[rating_key].values
        # use floats by default
        shape = (max(rows) + 1, max(columns) + 1)
        matrix = sps.csr_matrix((ratings, (rows, columns)), shape=shape)
        matrix.eliminate_zeros()
        return matrix

    def get_all_dataset (self):
        return self.URM