import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from app import cate_feat,cont_feat


def data_splitting(data):
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # noqa: E501
    return X_train, X_test, y_train, y_test


def scale_data(data_to_scale, cont_feat):
    data_cont = data_to_scale[cont_feat]
    scaler = StandardScaler()
    scaler.fit(data_cont)
    joblib.dump(scaler, '../models/scaler.joblib')
    loaded_scaler = joblib.load('../models/scaler.joblib')
    data_cont_scaled = pd.DataFrame(loaded_scaler.transform(data_cont), columns=data_cont.columns)  # noqa: E501

    return data_cont_scaled


def encode_data(data_to_encode, cate_feat, X_train_cate):

    data_cate = data_to_encode[cate_feat]
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(X_train_cate)
    joblib.dump(encoder, '../models/encoder.joblib')
    encoder = joblib.load('../models/encoder.joblib')
    data_cate_encoded = pd.DataFrame(encoder.transform(data_cate),
        columns=encoder.get_feature_names_out(data_cate.columns))  # noqa: E128,E501
    
    return data_cate_encoded

def build_model_for_train(X_train,y_train):
    
    X_train_cont_scaled = scale_data(X_train,cont_feat)
    X_train_cate = X_train[cate_feat]
    X_train_cate_encoded = encode_data(X_train,cate_feat,X_train_cate)
    X_train_preprocessed = pd.concat([X_train_cont_scaled, X_train_cate_encoded], axis=1)
    
    return X_train_preprocessed ,cate_feat  

def build_model_for_test(X_test,y_test,X_train_cate):

    X_test_cont_scaled = scale_data(X_test,cont_feat)
    X_test_cate_encoded = encode_data(X_test,cate_feat,X_train_cate)
    X_test_preprocessed = pd.concat([X_test_cont_scaled, X_test_cate_encoded], axis=1)
    
    return X_test_preprocessed

def model_train_and_predict(X_train_preprocessed,X_test_preprocessed,y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_preprocessed, y_train)
    y_test_pred = model.predict(X_test_preprocessed)

    return y_test_pred,model

def model_evaluation(y_test_pred,y_test,X_test_preprocessed,model):
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return test_mse,test_r2

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

    return round(rmsle, precision) 


def build_model(data: pd.DataFrame) -> dict[str, str]:
          
    X_train, X_test, y_train, y_test = data_splitting(data)

    X_train_preprocessed,cate_feat = build_model_for_train(X_train,y_train) 
    X_train_cate = X_train[cate_feat] 
    X_test_preprocessed = build_model_for_test(X_test,y_test,X_train_cate)
    y_test_pred,model = model_train_and_predict(X_train_preprocessed,X_test_preprocessed,y_train)
    test_mse,test_r2 = model_evaluation(y_test_pred,y_test,X_test_preprocessed,model)
    test_rmsle = compute_rmsle(y_test, y_test_pred)
    joblib.dump(model, '../models/model.joblib')   
    rmse_dict = {"rmse":test_rmsle}

    return rmse_dict['rmse']


