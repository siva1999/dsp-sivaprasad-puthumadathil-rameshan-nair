import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score

def model_analysis(data):
    data.head()
    print(f"\nshape of data {data.shape}")
    print(f"\ndata coloumns : {data.columns}")
    print(f"\nnull values in each features : {data.isnull().sum()}")
    print(f"\ndata types of each features : {data.dtypes}")    
    return
    
def data_splitting(train_data):
    '''
    now let's split the train_data set into train and test so that we can analyse 
    the performance of the model in an unbiased way
    '''

    # taking features into one dataframe and target into another dataframe

    X = train_data.drop('SalePrice', axis=1)
    y = train_data['SalePrice']

    # let's split the data - 80 % for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"shape after splitting : \n train data {X_train.shape} {y_train.shape} \n test data {X_test.shape} {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def feature_selection(data):
    # let's check which features are categorical and which are continuous
    print("continuous features :\n", data.select_dtypes(include=['int64']).dtypes.head())
    print("categorical features :\n", data.select_dtypes(include=['object']).dtypes.head())
    
    cate_feat = ['HouseStyle', 'Neighborhood', 'BldgType', 'KitchenQual', 'ExterQual'] 
    cont_feat = ['OverallQual', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
    
    print("The following features are picked")
    print("\n categorical features")
    for feat in cate_feat:
        print(feat)
    print("\n continuous features")
    for feat in cont_feat:
        print(feat)   
    
    print(type(cate_feat))
    
    return cate_feat,cont_feat

def checkformissingval_in_data(feat,curr_data):
    # handling missing values
        
    for val in feat:
        count = curr_data[val].isnull().sum()
        if(count != 0):
            print(f"Null values in feature {val} is {count}")
        else:
            print(f"No null value in feature {val}")

def scale_data(data_to_scale,cont_feat):
    #Scaling is a technique used to normalize the range of independent variables or features of data.
    # seperate the dataframe  into cont subset
    data_cont = data_to_scale[cont_feat]
    
    def findrange(data):
        min_values = data.min()
        max_values = data.max()

        feature_ranges = max_values - min_values

        print("Feature Minimums:\n", min_values)
        print("\nFeature Maximums:\n", max_values)
        print("\nFeature Ranges:\n", feature_ranges)
        
    findrange(data_cont)
    print(data_cont.head())
    print(f"\nthe shape of data cont is \n {data_cont.shape}")
    
    # scaling cont features
    scaler = StandardScaler()
    scaler.fit(data_cont)
    
    joblib.dump(scaler, '../models/scaler.joblib')
    
    loaded_scaler = joblib.load('../models/scaler.joblib')
    
    data_cont_scaled = pd.DataFrame(loaded_scaler.transform(data_cont), columns=data_cont.columns)
    
    print("\nlet's see how scaled data looks like:")
    print(data_cont_scaled.head())
    print("see if range is reduced or not")
    findrange(data_cont_scaled)
    print("shape of data scaled :",data_cont_scaled.shape)
    
    return data_cont_scaled


def encode_data(data_to_encode,cate_feat,X_train_cate):
    # Categorical features need to be encoded to numerical values.

    # seperate the dataframe  into categorical subset
    data_cate = data_to_encode[cate_feat]

    print(data_cate.head())

    print(f"\nthe shape of data to encode is \n {data_cate.shape}")
    
    #encoding the categorical values
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid dummy variable trap
    joblib.dump(encoder, '../models/encoder.joblib')
    
    loaded_encoder = joblib.load('../models/encoder.joblib')
    loaded_encoder.fit(X_train_cate)
    
    data_cate_encoded = pd.DataFrame(loaded_encoder.transform(data_cate),
                                    columns=loaded_encoder.get_feature_names_out(data_cate.columns))
    print(f"\nthe shape after encoding   \n {data_cate_encoded.shape}")
    
    print("see if the data is encoded or not")
    print(data_cate_encoded.head())
    
    return data_cate_encoded

def model_train_and_predict(X_train_preprocessed,X_test_preprocessed,y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_preprocessed, y_train)
    y_test_pred = model.predict(X_test_preprocessed)
    return y_test_pred,model
    
    
def build_model_for_train(X_train,y_train):
    
    print("\n==================== train data ============================")
    
    model_analysis(X_train)
        
    #feature selection
    cate_feat,cont_feat = feature_selection(X_train)
    
    #feature processing
    checkformissingval_in_data(cont_feat,X_train)
    checkformissingval_in_data(cate_feat,X_train)
    
    #scaling the continuous data
    X_train_cont_scaled = scale_data(X_train,cont_feat)
    
    X_train_cate = X_train[cate_feat]
    
    #encoding the categorical data
    X_train_cate_encoded = encode_data(X_train,cate_feat,X_train_cate)
    
    # Concatenate back the scaled and encoded features
    X_train_preprocessed = pd.concat([X_train_cont_scaled, X_train_cate_encoded], axis=1)
    
    return X_train_preprocessed ,cate_feat   

def build_model_for_test(X_test,y_test,X_train_cate):
    
    print("\n==================== test data ============================")
    print(f"\n shape after splitting test data : {X_test.shape} {y_test.shape}")
    model_analysis(X_test)
    
    #feature selection
    cate_feat,cont_feat = feature_selection(X_test)
    
    #feature processing
    checkformissingval_in_data(cont_feat,X_test)
    checkformissingval_in_data(cate_feat,X_test)
    
    # scaling the continuous data
    X_test_cont_scaled = scale_data(X_test,cont_feat)
    
    #encoding the categorical data
    X_test_cate_encoded = encode_data(X_test,cate_feat,X_train_cate)
    
    # Concatenate back the scaled and encoded features
    X_test_preprocessed = pd.concat([X_test_cont_scaled, X_test_cate_encoded], axis=1)
    
    return X_test_preprocessed

def model_evaluation(y_test_pred,y_test,X_test_preprocessed,model):
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")
    print(f"Test R^2: {test_r2}")
    print(f"model score is {model.score(X_test_preprocessed,y_test)}")

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)    
    
def build_model(data: pd.DataFrame) -> dict[str, str]:
    
    model_analysis(data)
    
    X_train, X_test, y_train, y_test = data_splitting(data)
           
    X_train_preprocessed,cate_feat = build_model_for_train(X_train,y_train)
    
    X_train_cate = X_train[cate_feat]
    
    X_test_preprocessed = build_model_for_test(X_test,y_test,X_train_cate)
    
    # train and predict the model
    y_test_pred,model = model_train_and_predict(X_train_preprocessed,X_test_preprocessed,y_train)
    
    #evaluate the model
    model_evaluation(y_test_pred,y_test,X_test_preprocessed,model)
    
    test_rmsle = compute_rmsle(y_test, y_test_pred)
    
    print(f"Test RMSLE: {test_rmsle}")
    
    # Persist the trained model
    joblib.dump(model, '../models/model.joblib')
    
    rmse_dict = {"rmse":test_rmsle}
    
    return rmse_dict,X_train_cate
    