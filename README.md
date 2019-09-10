# 빅콘테스트 Analysis 퓨처스리그.

---

**항공 운항 데이터를 활용한 “항공 지연 예측”**

- 항공 시즌 스케줄, 운항데이터 등 항공운항데이터(한국공항공사)와항공기상데이터 등을 활용하여 항공지연 예측 모형 개발을 통하여9월 16일부터 9월 30일까지의 항공편별 지연 여부 예측

### Team : 빅공 (이주형, 홍나단)

---

### Files

---

    |--- Data
    |    | -- AFSNT.csv, AFSNT_DLY.csv, SFSNT.csv 
    |--- Profiling
    |    | --- AFSNT(Train).html, AFSNT_DLY(Test).html
    |--functions.py
    |--Model.py
    |--main.py
    |--README.md

### Data

---

    AFSNT.csv (2017.1.1 ~ 2019.6.30 까지의 운항실적 데이터 : 학습 데이터)
    AFSNT_DLY.csv (2019.9.16 ~ 2019.9.30 까지 지연건수 및 지연확률 계산)
    SFSNT.csv (2019년 하계 스케줄 중 7월~9월이 포함된 시즌데이터)

### Reaquirements

---

- Python (3.x +)
- pandas
- numpy
- sklearn
- imbalanced-learn (sampling)
- xgboost

### Install

---
    python main.py
---

- main.py를 실행하면 전처리 부터 모델링까지 다 실행 가능. 
- 그러나 main.py에 있는 주석을 잘 보고 실행하기 바람.
- gridsearch는 시간이 매우 오래 걸리니 참고.

### Source Code.

---

1. functions.py
    - 전치리 및 새로운 변수를 추가해주는 함수들이 있는 소스 코드.

---
        def make_stt_weight(df):
            """
            Add STT_weight field fucntion
            variable_dic : key = 'hour' value =  that hour counts
        
            weight : hour counts / all data counts (list)
            """
            #code
        
        def make_stt_hour(df):
            """
            Add STT_HOUR field function
            """
            #code 
        
        def make_stt_term(df):
            """
            Add STT_term field function
            """
            #code 
        
        def make_quaruter(df):
            """
            Add QUARTER field function
            """
            #code 
        
        def make_one_hot_field(df, features):
            """
            categorical field(nominal) apply one-hot encoding.
            Add Dummy variable (one-hot)
            """
            #code
        
        def drop_field(df, features):
            """
            Drop fields
            """
            #code 
        
        def label_encoding(df, features):
            """
            ordinal field apply label encoding.
            Input features convert to label field
            :param df: input data frame
            :param features: to convert to label field
            :return: input data frame
            """
            #code
        
        def under_sampling(df):
            """
            Using TomekLinks Down Sampling
            """
            #code

---

2. Model.py
    - Xgboost, Adaboost, RandomForest, Logistic Regression 모형을 Customizing한 소스 코드.

---
        class name 
        : Xgb
        
        parameters & default value 
        : booster='gbtree', silent=True, min_child_weight=1, max_depth=6, gamma=0.1, alpha=0.01, colsample_bytree=1, colsample_bylever=1, n_estimators=50, nthread=4, objective='binary:logistic', random_state=2, sampled_type='weighted'
        
        instance variable
        - train_accrs : train data accuracy.
        - test_accrs : test data accuracy.(kflod validation.)
        - best_parmas : xgboost model best parameter.(gridsearch algorithm)
        - best_scores : xgboost model best scores.(gridsearch algorithm)
        - confusion_matrix : xgboost confusion matrix.
        - model : call xgboost model.
        
        functions 
        - train : trian data.(xgboost)
        > args : df(data), features, target_feature
        
        - gridSearch : search xgboost best parameter.
        > args : df(data), features, target_feature, param_grid
        
        prediect : predict test_df and add DLY field, DLY_RATE value
        
        
        class name 
        : RF
        
        instance variable
        - train_accrs : train data accuracy.
        - test_accrs : test data accuracy.(kflod validation.)
        
        - confusion_matrix : RandomForest confusion matrix.
        - model : call RandomForest model.
        
        functions 
        - train : trian data.(xgboost)
        > args : df(data), features, target_feature
        
        class name 
        : Adaboost
        
        instance variable
        - train_accrs : train data accuracy.
        - test_accrs : test data accuracy.(kflod validation.)
        
        - confusion_matrix : Adaboost confusion matrix.
        - model : call Adaboost model.
        
        functions 
        - train : trian data.(xgboost)
        > args : df(data), features, target_feature
        
        
        class name 
        : Logistic.
        
        instance variable
        - train_accrs : train data accuracy.
        - test_accrs : test data accuracy.(kflod validation.)
        
        - confusion_matrix : Logisitic Regression confusion matrix.
        - model : call Logisitic Regression model.
        
        functions 
        - train : trian data.(xgboost)
        > args : df(data), features, target_feature
---
3. main.py
    - 데이터 파이프라인 구축한 함수.
    - 모델 성능 확인하는 함수.(xgboost)
    - 최종 결과 파일 생성하는 파일.

    ### functions.

    1. set_up_df
        - 학습 전 전체 데이터 파이프 라인 구축.
            - 가변수 추가.
            - label encoding.
            - one-hot encoding.
    2. grid
        - xgboost 최적의 파라미터를 찾기 위한 Grid serach algorithm을 사용한 함수.
    3. compare_model
        - random forest, Adaboost, Logisitic Regresiion 모델 성능 출력 해주는 함수.
        - 각 모델들 성능 및 confusion matrix 확인하기 위함.