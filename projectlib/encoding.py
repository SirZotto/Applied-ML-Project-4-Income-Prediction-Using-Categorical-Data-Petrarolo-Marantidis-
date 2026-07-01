def drop_rows(df, drop_these_data_rows):
    """Drops all rows which include one of the values in drop_these_data_rows in any column.
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): just the dataset
        drop_these_rows (list): Drops all rows which include the strings which are in this list.
    """
    _df = df.copy()

    row_amount_before = len(_df)

    for column_name in _df.columns:
        mask = _df[column_name].isin(drop_these_data_rows)
        _df = _df[mask == False]

    amount_of_rows_deleted = row_amount_before - len(_df)

    _df = _df.reset_index(drop=True)

    print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie.{(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")

    return _df




####################################
def One_Hot_Encoder(df,column_to_be_encoded,number_if_True = 1, number_if_False = 0, delete_old_column = False,  drop_these_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates new column according to the columns categories.
       The rows then get marked wether the row has or has not the column property. 
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): just the dataset
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        number_if_True (int, optional): number if the column property applies to the row. Defaults to 1.
        number_if_False (int, optional): number if the column property does not apply to the row. Defaults to 0.
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_these_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()

    #row_amount_before = len(_df) # Optional if you want to print

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_these_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df = _df[~_df[column_name].isin( drop_these_data_rows)]

    #amount_of_rows_deleted = row_amount_before - len(_df) # Optional if you want to print

    _df = _df.reset_index(drop=True)

    row_amount = len(_df)
    
    for i,column_name in enumerate(column_to_be_encoded):
        list_of_categories = list(_df[column_name].unique()) #for each column we have a list of categories
        
        for category in list_of_categories:
            new_column_name = column_name +"_"+ category
            _df[new_column_name] = (_df[column_name] == category).map({True: number_if_True, False: number_if_False})
                
        if delete_old_column:
            _df = _df.drop(columns=[column_name])
    
    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie. {(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")     # Optional if you want to print        
    return _df




##############################################
def Ordinal_Encoder(df,column_to_be_encoded, categories_order =[[]], starting_numbers = None, step_sizes= None, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates transforms the columns accordingly to the other inputs. 
       outputs then a new edited copy of df. Important: the old columns get overridden with the new values (in the copy)

    Args:
        df (pd.dataframe): just the dataset
        column_to_be_encoded (list): a list of column names from df which need to be encoded
        categories_order (list of list): the length of the outer list is the same as column_to_be_encoded, the inner list give the categories AND the order of these categories MUST be INCREASING. if a category is given as a list list itself, fe [cat1,cat2] they will be set equal
        starting_numbers (list of doubles, optional): list of numbers with the same length as column_to_be_encoded, each number corresponds to the first entry of each inner list of categories_order. Defaults to starting each order with 1
        step_sizes (list of doubles, optional): list of numbers with the same length as column_to_be_encoded, each number corresponds to the step size of each inner list of categories_order. Defaults to step size 1
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()

    #row_amount_before = len(_df) # Optional if you want to print 

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            mask = _df[column_name].isin(drop_unknown_data_rows)
            _df = _df[mask == False]

    #amount_of_rows_deleted = row_amount_before - len(_df)  # Optional if you want to print 

    _df = _df.reset_index(drop=True)

    #row_amount = len(_df)  # Optional if you want to print 

    if starting_numbers is None:
        starting_numbers = [1 for _ in range(len(column_to_be_encoded))]

    if step_sizes is None:
        step_sizes = [1 for _ in range(len(column_to_be_encoded))]
        
    #handle errors here
    if len(column_to_be_encoded) != len(categories_order):
        raise ValueError("len(column_to_be_encoded) must be equal to len(categories_order)")

    if len(column_to_be_encoded) != len(starting_numbers):
        raise ValueError("len(column_to_be_encoded) must be equal to len(starting_numbers)")

    if len(column_to_be_encoded)!= len(step_sizes):
        raise ValueError("len(column_to_be_encoded) must be equal to len(step_sizes)")

    for i in range(len(column_to_be_encoded)):
        column_name = column_to_be_encoded[i]
        current_number = starting_numbers[i]
        replacement_dict = {}

        for category_or_list in categories_order[i]:
            if type(category_or_list) == list:
                for category in category_or_list:
                    replacement_dict[category] = current_number
            else:
                replacement_dict[category_or_list] = current_number

            current_number = current_number +step_sizes[i]

        old_column = _df[column_name].copy( )
        _df[column_name]= old_column.map(replacement_dict)

        if _df[column_name].isna().any():
            categories_not_in_order = list(old_column[_df[column_name].isna()].unique())
            raise ValueError(f"these categories from column '{column_name}' were not found in categories_order: {categories_not_in_order}")

    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie.{100*row_amount/row_amount_before}% still remain")  # Optional if you want to print 

    return _df




#############

def Frequency_Encoder(df,column_to_be_encoded, delete_old_column = False, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates new column called "{oldcolumnname}_freq".
       The new column has then the frequency representation
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): just the dataset
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()

    #row_amount_before = len(_df)  # Optional if you want to print 

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df = _df[~_df[column_name].isin(drop_unknown_data_rows)]

    #amount_of_rows_deleted = row_amount_before - len(_df)  # Optional if you want to print 

    _df = _df.reset_index(drop=True)

    row_amount = len(_df)
    
    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]
    
    for i,column_name in enumerate(column_to_be_encoded):
        
        new_column_name = f"{column_name}_freq"
        counting_dataframe =_df[column_name].value_counts().to_frame().T.reset_index(drop=True) # gives me every catgory and their respective amount with col names of the categories and the 0-th line their respective amount
        
        _df[new_column_name] = _df[column_name].map(counting_dataframe.loc[0] / row_amount)
        
    if delete_old_column:
        _df = _df.drop(columns=column_to_be_encoded)
        
    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie.{(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")  # Optional if you want to print           
    return _df









############

def Target_Encoder(df,column_to_be_encoded,target_column_name, delete_old_column = False, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates new column called "{oldcolumnname}_target".
       The new column has then the target representation
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): just the dataset
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        target_column_name (str): the column name of the target variable
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()
    #row_amount_before = len(_df) # Optional if you want to print 

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df = _df[~_df[column_name].isin(drop_unknown_data_rows)]

    #amount_of_rows_deleted = row_amount_before - len(_df) # Optional if you want to print 

    _df = _df.reset_index(drop=True)
    for i,column_name in enumerate(column_to_be_encoded):
        
        new_column_name = f"{column_name}_target"
        counting_dataframe = _df.groupby(column_name)[target_column_name].mean().to_frame().T.reset_index(drop=True) # gives me every catgory and their respective target mean with col names of the categories and the 0-th line their respective target mean
        
        _df[new_column_name] = _df[column_name].map(counting_dataframe.loc[0])
        
    if delete_old_column:
        _df = _df.drop(columns=column_to_be_encoded)
        
    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie. {(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")  # Optional if you want to print           
    return _df

def Target_Encoder_LC(df_to_encode,df_for_target_statistics,column_to_be_encoded,target_column_name, delete_old_column = False, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates new column called "{oldcolumnname}_target".
       The new column has then the target representation, but the target statistics are taken from df_for_target_statistics
       outputs then a new edited copy of df_to_encode.
       (Basically just a special version of Target_Encoder_LC)
       
    Args:
        df_to_encode (pd.dataframe): just the dataset which needs to be encoded
        df_for_target_statistics (pd.dataframe): the dataset oh which the data is used to encode for df_to_encode
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        target_column_name (str): the column name of the target variable
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df_to_encode = df_to_encode.copy()
    _df_for_target_statistics = df_for_target_statistics.copy()

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df_to_encode = _df_to_encode[~_df_to_encode[column_name].isin(drop_unknown_data_rows)]
            _df_for_target_statistics = _df_for_target_statistics[~_df_for_target_statistics[column_name].isin(drop_unknown_data_rows)]

    _df_to_encode = _df_to_encode.reset_index(drop=True)
    _df_for_target_statistics = _df_for_target_statistics.reset_index(drop=True)

    global_target_mean = _df_for_target_statistics[target_column_name].mean()

    for i,column_name in enumerate(column_to_be_encoded):
        
        new_column_name = f"{column_name}_target"
        counting_dataframe = _df_for_target_statistics.groupby(column_name)[target_column_name].mean().to_frame().T.reset_index(drop=True)
        
        _df_to_encode[new_column_name] = _df_to_encode[column_name].map(counting_dataframe.loc[0])
        _df_to_encode[new_column_name] = _df_to_encode[new_column_name].fillna(global_target_mean)
        
    if delete_old_column:
        _df_to_encode = _df_to_encode.drop(columns=column_to_be_encoded)
        
    return _df_to_encode



#######


def SmoothTarget_Encoder(df,column_to_be_encoded,target_column_name, alpha = 1.0, delete_old_column = False, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and columns which need to be encoded and creates new column called "{oldcolumnname}_smooth_target".
       The new column has then the smoothed target representation
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): just the dataset
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        target_column_name (str): the column name of the target variable
        alpha (double, optional): smoothing parameter from the formula. Defaults to 1.0.
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()
    #row_amount_before = len(_df) # Optional if you want to print 

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df = _df[~_df[column_name].isin(drop_unknown_data_rows)]

    #amount_of_rows_deleted = row_amount_before - len(_df) # Optional if you want to print 

    _df = _df.reset_index(drop=True)

    global_target_mean = _df[target_column_name].mean()
    
    for i,column_name in enumerate(column_to_be_encoded):
        
        new_column_name = f"{column_name}_smooth_target"
        grouped_dataframe = _df.groupby(column_name)[target_column_name].agg(["mean","count"]) # gives me every catgory and their respective target mean and amount
        
        smooth_target_series = (grouped_dataframe["count"] * grouped_dataframe["mean"] + alpha * global_target_mean) / (grouped_dataframe["count"] + alpha)
        
        _df[new_column_name] = _df[column_name].map(smooth_target_series)
        
    if delete_old_column:
        _df = _df.drop(columns=column_to_be_encoded)
        
    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie. {(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")  # Optional if you want to print           
    return _df


def SmoothTarget_Encoder_LC(df,df_for_target_statistics,column_to_be_encoded,target_column_name, alpha = 1.0, delete_old_column = False, drop_unknown_data_rows = [] ):
    """takes a pandas dataframe and a second pandas dataframe from which the target statistics are computed and creates new column called "{oldcolumnname}_smooth_target".
       The new column has then the smoothed target representation
       outputs then a new edited copy of df.

    Args:
        df (pd.dataframe): the dataset which gets encoded
        df_for_target_statistics (pd.dataframe): the dataset from which the target statistics are computed
        column_to_be_encoded (list): a list of column names form df which need to be encoded
        target_column_name (str): the column name of the target variable
        alpha (double, optional): smoothing parameter from the formula. Defaults to 1.0.
        delete_old_column (bool, optional): Wether the columns form column_to_be_encoded are deleted or not. Defaults to False.
        drop_unknown_data_rows (list, optional): Drops all rows which include the strings which are in this list. Defaults to empty list.
    """
    _df = df.copy()
    _df_for_target_statistics = df_for_target_statistics.copy()
    #row_amount_before = len(_df) # Optional if you want to print 

    if type(column_to_be_encoded) == str:
        column_to_be_encoded = [column_to_be_encoded]

    if drop_unknown_data_rows !=[]:
        for column_name in column_to_be_encoded:
            _df = _df[~_df[column_name].isin(drop_unknown_data_rows)]
            _df_for_target_statistics = _df_for_target_statistics[~_df_for_target_statistics[column_name].isin(drop_unknown_data_rows)]

    #amount_of_rows_deleted = row_amount_before - len(_df) # Optional if you want to print 

    _df = _df.reset_index(drop=True)
    _df_for_target_statistics = _df_for_target_statistics.reset_index(drop=True)

    global_target_mean = _df_for_target_statistics[target_column_name].mean()
    
    for i,column_name in enumerate(column_to_be_encoded):
        
        new_column_name = f"{column_name}_smooth_target"
        grouped_dataframe = _df_for_target_statistics.groupby(column_name)[target_column_name].agg(["mean","count"]) # gives me every catgory and their respective target mean and amount
        
        smooth_target_series = (grouped_dataframe["count"] * grouped_dataframe["mean"] + alpha * global_target_mean) / (grouped_dataframe["count"] + alpha)
        
        _df[new_column_name] = _df[column_name].map(smooth_target_series)
        _df[new_column_name] = _df[new_column_name].fillna(global_target_mean)
        
    if delete_old_column:
        _df = _df.drop(columns=column_to_be_encoded)
        
    #print(f"{amount_of_rows_deleted} out of {row_amount_before} were deleted, ie. {(1- amount_of_rows_deleted/row_amount_before)*100}% still remain ")  # Optional if you want to print           
    return _df