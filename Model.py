from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np


class Xgb:

    def __init__(self, booster='gbtree',  min_child_weight=1, max_depth=6, gamma=0.1, \
                 alpha=0.01, colsample_bytree=1, colsample_bylevel=1, n_estimators=50, nthread=4,\
                 objective='binary:logistic', random_state=2):
        
        self.train_accrs = []
        self.test_accrs = []
        
        self.best_params = []
        self.best_scores = []
        
        self.confusion_matrixs = []

        self.model = XGBClassifier(booster=booster, alpha=alpha, objective=objective, max_depth=max_depth, gamma=gamma,\
                                   min_child_weight=min_child_weight, colsample_bylevel=colsample_bylevel, colsample_bytree=colsample_bytree, \
                                   random_state=random_state, n_estimators=n_estimators, nthread=nthread)

    def train(self, df, features, target_feature):
        """
        Using model selection kFold
        """
        # if new train clear self.train_accrs, self.test_accrs clear
        df = df.astype(float)
        
        self.train_accrs.clear()
        self.test_accrs.clear()
        self.confusion_matrixs.clear()

        fold_idx = 1
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(df):
            print('[Fold {}] train size = {}, test size = {}'.format(fold_idx, len(train_index), len(test_index)))
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]

            train_y = train_data[target_feature]
            train_x = train_data[features]

            test_y = test_data[target_feature]
            test_x = test_data[features]

            model = self.model # 파라미터 설정하기
            model.fit(train_x, train_y)

            self.test_accrs.append(model.score(test_x, test_y))
            self.train_accrs.append(model.score(train_x, train_y))

            fold_idx += 1

            # check confusion matrix
            pred_y = model.predict(test_x)
            self.confusion_matrixs.append(confusion_matrix(test_y, pred_y))

        print("=========train_mean_accuracy=========")
        print(np.average(self.train_accrs))

        print("=========test_mean_accuracy==========")
        print(np.average(self.test_accrs))

        print("=========confusion_matrix============")
        print(self.confusion_matrixs[0])        
        
        return

    def predict(self, train_df, test_df, features, target_feature):
        train_x = train_df[features]
        test_x = test_df[features]

        model = self.model
        model.fit(train_x, train_df[target_feature])

        pred_y = model.predict(test_x)
        pred_y_rate = model.predict_proba(test_x)

        test_df['DLY'] = pred_y
        test_df['DLY_RATE'] = pred_y_rate

        return test_df

    def gridSearch(self, df, features, target_feature, param_grid):
        """
        Search best parameter in Xgboost algorithm.

        Searching Params
            {'booster'
            'silent'
            'max_depth'
            'min_child_weight'
            'gamma'
            'nthread'
            'colsample_bytree'
            'colsample_bylevel'
            'n_estimators'
            'objective'
            'random_state'}
        """
        self.best_params.clear()
        self.best_scores.clear()

        # kf = KFold(n_splits=2, shuffle=True)
        model = XGBClassifier()

        #start grid search
        gcv = GridSearchCV(model, param_grid=param_grid, scoring='f1', n_jobs=4)
        gcv.fit(df[features].values, df[target_feature].values)

        self.best_params.append(gcv.best_params_)
        self.best_scores.append(gcv.best_score_)

        return


class RF:


    def __init__(self):

        self.train_accrs = []
        self.test_accrs = []

        self.confusion_matrixs = []
        self.model = RandomForestClassifier()

    def train(self, df, features, target_feature):
        """
        Using model selection kFold
        """
        # if new train clear self.train_accrs, self.test_accrs clear
        df = df.astype(float)

        self.train_accrs.clear()
        self.test_accrs.clear()
        self.confusion_matrixs.clear()

        fold_idx = 1
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(df):
            print('[Fold {}] train size = {}, test size = {}'.format(fold_idx, len(train_index), len(test_index)))
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]

            train_y = train_data[target_feature]
            train_x = train_data[features]

            test_y = test_data[target_feature]
            test_x = test_data[features]

            model = self.model  # 파라미터 설정하기
            model.fit(train_x, train_y)

            self.test_accrs.append(model.score(test_x, test_y))
            self.train_accrs.append(model.score(train_x, train_y))

            fold_idx += 1

            # check confusion matrix
            pred_y = model.predict(test_x)
            self.confusion_matrixs.append(confusion_matrix(test_y, pred_y))

        print("=========train_mean_accuracy=========")
        print(np.average(self.train_accrs))

        print("=========test_mean_accuracy==========")
        print(np.average(self.test_accrs))

        print("=========confusion_matrix============")
        print(self.confusion_matrixs[0])

        return


class AdaBoost:


    def __init__(self):
        self.train_accrs = []
        self.test_accrs = []

        self.confusion_matrixs = []

        self.model = AdaBoostClassifier()

    def train(self, df, features, target_feature):
        """
        Using model selection kFold
        """
        # if new train clear self.train_accrs, self.test_accrs clear
        df = df.astype(float)

        self.train_accrs.clear()
        self.test_accrs.clear()
        self.confusion_matrixs.clear()

        fold_idx = 1
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(df):
            print('[Fold {}] train size = {}, test size = {}'.format(fold_idx, len(train_index), len(test_index)))
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]

            train_y = train_data[target_feature]
            train_x = train_data[features]

            test_y = test_data[target_feature]
            test_x = test_data[features]

            model = self.model  # 파라미터 설정하기
            model.fit(train_x, train_y)

            self.test_accrs.append(model.score(test_x, test_y))
            self.train_accrs.append(model.score(train_x, train_y))

            fold_idx += 1

            # check confusion matrix
            pred_y = model.predict(test_x)
            self.confusion_matrixs.append(confusion_matrix(test_y, pred_y))

        print("=========train_mean_accuracy=========")
        print(np.average(self.train_accrs))

        print("=========test_mean_accuracy==========")
        print(np.average(self.test_accrs))

        print("=========confusion_matrix============")
        print(self.confusion_matrixs[0])

        return


class Logistic:


    def __init__(self):
        self.train_accrs = []
        self.test_accrs = []

        self.confusion_matrixs = []

        self.model = LogisticRegression()

    def train(self, df, features, target_feature):
        """
        Using model selection kFold
        """
        # if new train clear self.train_accrs, self.test_accrs clear
        df = df.astype(float)

        self.train_accrs.clear()
        self.test_accrs.clear()
        self.confusion_matrixs.clear()

        fold_idx = 1
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(df):
            print('[Fold {}] train size = {}, test size = {}'.format(fold_idx, len(train_index), len(test_index)))
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]

            train_y = train_data[target_feature]
            train_x = train_data[features]

            test_y = test_data[target_feature]
            test_x = test_data[features]

            model = self.model  # 파라미터 설정하기
            model.fit(train_x, train_y)

            self.test_accrs.append(model.score(test_x, test_y))
            self.train_accrs.append(model.score(train_x, train_y))

            fold_idx += 1

            # check confusion matrix
            pred_y = model.predict(test_x)
            self.confusion_matrixs.append(confusion_matrix(test_y, pred_y))

        print("=========train_mean_accuracy=========")
        print(np.average(self.train_accrs))

        print("=========test_mean_accuracy==========")
        print(np.average(self.test_accrs))

        print("=========confusion_matrix============")
        print(self.confusion_matrixs[0])

        return








    
  
