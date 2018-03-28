import pandas as pd
import numpy as np
import scipy.sparse as sps


def load_dataframes(path, sep, drop_columns=[], implicit_field=None, implicit_treshold=3, names=None, header='infer'):
    df = pd.read_csv(path, sep=sep, header=header, names=names)
    if len(drop_columns) > 0:
        df.drop(drop_columns, axis=1, inplace=True)
    if implicit_field is not None:
        df[implicit_field] = df[implicit_field].apply(lambda x: 0 if x < implicit_treshold else 1)
    return df
