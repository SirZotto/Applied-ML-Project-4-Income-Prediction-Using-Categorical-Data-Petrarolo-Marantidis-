import numpy as np
from courselib.utils.splits import train_test_split


def k_fold_split_dataframe(df,k):
    perm_ind = np.random.permutation(np.arange(df.shape[0]))
    return [df.iloc[fold_ind] for fold_ind in np.array_split(perm_ind,k)]
    
def data_to_numpy_transfomer(df,class_column_name='Class'):
    X, Y, X_train, Y_train, X_test, Y_test =  train_test_split(df,training_data_fraction=1,class_column_name=class_column_name,return_numpy=True, shuffle= False)
    return X, Y