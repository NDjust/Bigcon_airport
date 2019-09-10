import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler



def make_stt_weight(df):
    """
    Add STT_weight field fucntion
    variable_dic : key = 'hour' value =  that hour counts

    weight : hour counts / all data counts (list)
    """
    variable_dic = {}
    weight = []

    # make variable dictionary
    for i in range(len(df.STT_HOUR.value_counts().index)):
        variable_dic[str(df.STT_HOUR.value_counts().index[i])] = df.STT_HOUR.value_counts().values[i]

    for hour in df.STT_HOUR.values:
        weight.append(variable_dic[str(hour)] / len(df))

    df['STT_weight'] = weight

    return df


def make_stt_hour(df):
    """
    Add STT_HOUR field function
    """
    hours = []

    for i in range(len(df)):
        index = df.STT.values[i].index(':')
        hours.append(int(df.STT.values[i][:index]))

    df['STT_HOUR'] = hours

    return df


def make_stt_term(df):
    """
    Add STT_term field function
    """
    time_term = []

    for i in range(len(df)):
        hour = df.STT_HOUR.values[i]
        if 0 <= hour < 7:
            time_term.append("dawn")
        elif 7 <= hour < 12:
            time_term.append("morning")
        elif 12 <= hour < 16:
            time_term.append('afternoon')
        elif 16 <= hour < 20:
            time_term.append("peak")
        else:
            time_term.append("evening")

    return df


def make_quaruter(df):
    """
    Add QUARTER field function
    """
    quarters = []

    for i in range(len(df)):
        if df["SDT_MM"][i] in (1, 2, 3):
            quarters.append("First")
        elif df["SDT_MM"][i] in (4, 5, 6):
            quarters.append("Second")
        elif df["SDT_MM"][i] in (7, 8, 9):
            quarters.append("Third")
        else:
            quarters.append("Fourth")

    df["QUARTER"] = quarters

    return df


def make_one_hot_field(df, features):
    """
    categrical field(nominal) apply one-hot encoding.
    Add Dummy variable (one-hot)
    """
    for feature in features:
        oec = OneHotEncoder()
        X = oec.fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        df_onehot = pd.DataFrame(X, columns=[feature + "_" + str(int(i)) for i in range(X.shape[1])])
        df = pd.concat([df, df_onehot], axis=1)

    return df


def drop_field(df, features):
    """
    Drop fields
    """
    df = df.drop(features, axis=1)
    
    return df


def label_encoding(df, features):
    """
    ordinal field apply label encoding.
    Input features convert to label field
    :param df: input data frame
    :param features: to convert to label field
    :return: None
    """
    for feature in features:
        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

    return df

def under_sampling(df):
    """
    Using Random Sampling
    """

    rus  = RandomUnderSampler(return_indices=True)
    X_tl, y_tl, id_tl = rus.fit_sample(df, df['DLY'])

    # remake data frame.
    columns = df.columns
    df = pd.DataFrame(X_tl, columns=columns)
    # df = df.astype(float)

    return df
