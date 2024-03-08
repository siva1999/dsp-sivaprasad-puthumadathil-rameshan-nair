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

def imputation(test_data):
    test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].median(), inplace=True)
    test_data['GarageArea'].fillna(test_data['GarageArea'].median(), inplace=True)
    test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0], inplace=True)
    
def make_predictions(test_data: pd.DataFrame,X_train) -> np.ndarray:
    
    model_analysis(test_data)
    
    cate_feat,cont_feat = feature_selection(test_data)
    
    print(f"cont_values selected are : {cont_feat} \ncate feat selected are {cate_feat}")
    
    #feature processing
    checkformissingval_in_data(cont_feat,test_data)
    checkformissingval_in_data(cate_feat,test_data)
    
    #since there are missing values,perform imputation
    
    imputation(test_data)
    print("\nafter doing imputation the missing value status are :\n")
    checkformissingval_in_data(cont_feat,test_data)
    checkformissingval_in_data(cate_feat,test_data)
    
    # seperate the dataframe  into cont subset
    test_cont = test_data[cont_feat]
    test_cate = test_data[cate_feat]
    
    print(test_cont.columns)
    
    # scaling the continuous data
    loaded_scaler = joblib.load('../models/scaler.joblib')
    test_cont_scaled = pd.DataFrame(loaded_scaler.transform(test_cont), columns=test_cont.columns)
    print("\n",test_cont_scaled.head())
    
    
    #encoding categorical values
    X_train_cate = X_train[cate_feat]
    
    loaded_encoder = joblib.load('../models/encoder.joblib')
    loaded_encoder.fit(X_train_cate)
    
    test_cate_encoded = pd.DataFrame(loaded_encoder.transform(test_cate),
                                   columns=loaded_encoder.get_feature_names_out(test_cate.columns))
    print(f"\nthe shape after encoding X_test_cate_encoded is {test_cate_encoded.shape}")
    
    test_preprocessed = pd.concat([test_cont_scaled, test_cate_encoded], axis=1)
    print(test_preprocessed.head())
    
    loaded_model = joblib.load('../models/model.joblib')
    test_pred = loaded_model.predict(test_preprocessed)
    
    print("The first 10 predicted SalePrice values of the test set are :\n",test_pred[:10])
    
make_predictions(test_data,X_train)