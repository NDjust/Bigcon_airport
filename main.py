from function import *
from Model import *
from collections import OrderedDict

import numpy as np
import pandas as pd

train_datapath = './data/AFSNT.csv'
test_datapath = './data/AFSNT_DLY.csv'

train_df = pd.read_csv(train_datapath, encoding='cp949')
test_df = pd.read_csv(test_datapath, encoding='cp949')

train_airport_dic = OrderedDict()
test_airport_dic = OrderedDict()


def set_up_df(df, df_name='train'):
    df_dic = OrderedDict()

    features = ['REG', 'IRR', 'ATT', 'DRR', 'CNL', 'CNR']

    # df.drop(features, axis = 1)
    for feature in features:
        if feature in df.columns:
            df = drop_field(df, features)

    # add feature
    df = make_stt_hour(df)  # should start make_stt_hour function.
    df = make_stt_term(df)
    df = make_quaruter(df)

    # one-hot Aod field
    one_hot_features = ['AOD']
    df = make_one_hot_field(df, one_hot_features)

    # drop AOD field
    df = drop_field(df, ['AOD'])

    # split data (Each airport)
    airport_list = []
    if df_name == "train":
        airport_list = ['A', 'B', 'F', 'H', 'I', 'J', 'L']
        df_dic['M'] = df  # test data "M" airport replace all_airport
    else:
        airport_list = ['A', 'B', 'F', 'H', 'I', 'J', 'L', 'M']

    for airport in airport_list:
        print("Split {} airport".format(airport))
        df_dic[airport] = df[df['FLO'] == airport]

    # Add STT_weight filed
    if df_name == "train":
        df_dic['M'] = make_stt_weight(df_dic['M'])

    for airport in df_dic.keys():
        print("Add {} airport STT_weight Field".format(airport))
        df_dic[airport] = make_stt_weight(df_dic[airport])

    features = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT',
                'STT', 'DLY', 'STT_HOUR', 'QUARTER', 'AOD_0', 'AOD_1']

    # Apply label encoding
    for airport in df_dic.keys():
        print("Apply to {} airport label encoding".format(airport))
        df_dic[airport] = label_encoding(df_dic[airport], features=features)

    # Apply down sampling
    if df_name == "train":
        for airport in df_dic.keys():
            print("Apply to {} airport down sampling".format(airport))
            df_dic[airport] = under_sampling(df_dic[airport])

    if df_name == "train":
        global train_airport_dic
        train_airport_dic = df_dic.copy()
    else:
        global test_airport_dic
        test_airport_dic = df_dic.copy()

    print(df_dic['M'].head())
    return


def grid():
    xgb = Xgb()
    param_grid = {'booster': ['gbtree'],
                  'max_depth': [7, 8, 9],
                  'min_child_weight': [1, 2, 3, 5],
                  'gamma': [0, 0.01, 0.03],
                  'alpha': [0.01, 0.03, 0.05],
                  'colsample_bytree': [0.6, 0.8, 0.9],
                  'colsample_bylevel': [0.6, 0.8, 0.9],
                  'n_estimators': [50, 30],
                  'objective': ['binary:logistic'],
                  'random_state': [2]}

    # remove 'DLY', 'STT' field
    features = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT',
                'STT_HOUR', 'QUARTER', 'AOD_0', 'AOD_1', 'STT_weight']

    for airport in train_airport_dic.keys():
        print("Start {} airport GridSearch".format(airport))
        
        data_frame = train_airport_dic[airport]
        xgb.gridSearch(data_frame, features=features, target_feature='DLY', param_grid=param_grid)    
        
        # save models best parameter by using gridSearch. 
        with open("./data/models_parameter.txt", 'a') as f:
            print("Start to Save {} airport Model best parameter".format(airport))

            f.write("============{}==========\n".format(airport))
            f.write("best params : {}\n".format(xgb.best_params))
            f.write("best score : {}\n".format(xgb.best_scores))

            print("Finish Save {} airport Model best Parameter".format(airport))

    return


def test_modeling():
    xgb = Xgb(min_child_weight=2, max_depth=9,\
                                   gamma=0, colsample_bytree=0.9, colsample_bylevel=0.9, n_estimators=50,\
                                   objective='binary:logistic', random_state=2, nthread=4, alpha=0.01)

    features = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT',
                'STT_HOUR', 'QUARTER', 'AOD_0', 'AOD_1', 'STT_weight']

    for airport in train_airport_dic.keys():
        print("Start {} Airport Training".format(airport))
        xgb.train(train_airport_dic[airport], features=features, target_feature='DLY')

    return


def result():
    xgb = Xgb(min_child_weight=2, max_depth=9, \
              gamma=0, colsample_bytree=0.9, colsample_bylevel=0.9, n_estimators=50, \
              objective='binary:logistic', random_state=2, nthread=4, alpha=0.01)

    features = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT',
                'STT_HOUR', 'QUARTER', 'AOD_0', 'AOD_1', 'STT_weight']

    global test_airport_dic
    global train_airport_dic

    for airport in test_airport_dic.keys():
        train_df = train_airport_dic[airport]
        test_df = test_airport_dic[airport]

        test_airport_dic[airport] = xgb.predict(train_df, test_df, features, target_feature='DLY')
        print("=========={} airport result===============".format(airport))
        test_airport_dic[airport].to_csv("./data/result_{}.csv".format(airport), encoding='cp949')

    return


def compare_model():
    rf = RF()
    adaboost = AdaBoost()
    logistic = Logistic()

    global train_airport_dic

    features = ['SDT_YY', 'SDT_MM', 'SDT_DD', 'SDT_DY', 'ARP', 'ODP', 'FLO', 'FLT',
                'STT_HOUR', 'QUARTER', 'AOD_0', 'AOD_1', 'STT_weight']

    df = train_airport_dic['M']
    print("======RF=========")
    rf.train(df, features=features, target_feature='DLY')

    print()
    print("=========Adaboost=======")
    adaboost.train(df, features=features, target_feature='DLY')

    print()
    print("=========logistic=======")
    logistic.train(df, features=features, target_feature='DLY')


if __name__ == "__main__":
    set_up_df(train_df, df_name='train')
    set_up_df(test_df, df_name='test')

    # compare_model()
    # grid() Too long time. Don't remove #.
    # test_modeling() # Testing modeling(Xgb model)

    result()

