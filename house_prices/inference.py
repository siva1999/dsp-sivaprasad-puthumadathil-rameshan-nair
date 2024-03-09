import pandas as pd
import numpy as np
import joblib

# read test data
test_data = pd.read_csv('../data/house-prices-advanced-regression-techniques/test.csv')   # noqa: E501


def model_analysis(data):
    data.head()
    print(f"\nshape of data {data.shape}")
    print(f"\ndata coloumns : {data.columns}")
    print(f"\nnull values in each features : {data.isnull().sum()}")
    print(f"\ndata types of each features : {data.dtypes}")
    return


def feature_selection(data):
    # let's check which features are categorical and which are continuous
    print("cont feat :\n", data.select_dtypes(include=['int64']).dtypes.head())
    print("cat feat :\n", data.select_dtypes(include=['object']).dtypes.head())

    cate_feat = ['HouseStyle', 'Neighborhood', 'BldgType', 'KitchenQual', 'ExterQual']  # noqa: E501
    cont_feat = ['OverallQual', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']  # noqa: E501

    print("The following features are picked")
    print("\n categorical features")
    for feat in cate_feat:
        print(feat)
    print("\n continuous features")
    for feat in cont_feat:
        print(feat)

    print(type(cate_feat))

    return cate_feat, cont_feat


def checkformissingval_in_data(feat, curr_data):
    # handling missing values

    for val in feat:
        count = curr_data[val].isnull().sum()
        if count != 0:
            print(f"Null values in feature {val} is {count}")
        else:
            print(f"No null value in feature {val}")


def imputation(test_data):
    test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].median(), inplace=True)  # noqa: E501
    test_data['GarageArea'].fillna(test_data['GarageArea'].median(), inplace=True)  # noqa: E501
    test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0], inplace=True)  # noqa: E501


def make_predictions(test_data: pd.DataFrame, X_train_cate) -> np.ndarray:

    model_analysis(test_data)

    cate_feat, cont_feat = feature_selection(test_data)

    print(f"cont_val: {cont_feat} \ncate feat : {cate_feat}")

    # feature processing
    checkformissingval_in_data(cont_feat, test_data)
    checkformissingval_in_data(cate_feat, test_data)

    # since there are missing values, perform imputation
    imputation(test_data)
    print("\nafter doing imputation the missing value status are :\n")
    checkformissingval_in_data(cont_feat, test_data)
    checkformissingval_in_data(cate_feat, test_data)

    # separate the dataframe into cont subset
    test_cont = test_data[cont_feat]
    test_cate = test_data[cate_feat]

    print(test_cont.columns)

    # scaling the continuous data
    loaded_scaler = joblib.load('../models/scaler.joblib')
    test_cont_scaled = pd.DataFrame(loaded_scaler.transform(test_cont), columns=test_cont.columns)  # noqa: E501
    print("\n", test_cont_scaled.head())

    loaded_encoder = joblib.load('../models/encoder.joblib')
    loaded_encoder.fit(X_train_cate)

    test_cate_encoded = pd.DataFrame(loaded_encoder.transform(test_cate),
                                     columns=loaded_encoder.get_feature_names_out(test_cate.columns))  # noqa: E501
    print(f"\nthe shape after encoding is {test_cate_encoded.shape}")

    test_preprocessed = pd.concat([test_cont_scaled, test_cate_encoded], axis=1)  # noqa: E501
    print(test_preprocessed.head())

    loaded_model = joblib.load('../models/model.joblib')
    test_pred = loaded_model.predict(test_preprocessed)

    print("The first 10 predicted SalePrice values are :\n", test_pred[:10])


make_predictions(test_data)
