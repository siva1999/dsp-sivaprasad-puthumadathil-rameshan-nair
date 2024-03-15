import pandas as pd
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


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
    joblib.dump(encoder, '../models/encoder.joblib')
    loaded_encoder = joblib.load('../models/encoder.joblib')
    loaded_encoder.fit(X_train_cate)
    data_cate_encoded = pd.DataFrame(loaded_encoder.transform(data_cate),
        columns=loaded_encoder.get_feature_names_out(data_cate.columns))  # noqa: E128,E501
    
    return data_cate_encoded

def feature_selection(data):
    cate_feat = ['HouseStyle', 'Neighborhood', 'BldgType', 'KitchenQual', 'ExterQual']  # noqa: E501
    cont_feat = ['OverallQual', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']  # noqa: E501
    return cate_feat, cont_feat

def imputation(test_data):
    # for replacing the null values of data
    test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].median(), inplace=True)  # noqa: E501
    test_data['GarageArea'].fillna(test_data['GarageArea'].median(), inplace=True)  # noqa: E501
    test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0], inplace=True)  # noqa: E501
