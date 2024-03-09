import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def model_analysis(data):
    data.head()
    print(f"\nshape of data {data.shape}")
    print(f"\ndata coloumns : {data.columns}")
    print(f"\nnull values in each features : {data.isnull().sum()}")
    print(f"\ndata types of each features : {data.dtypes}")
    return


def data_splitting(train_data):
    '''
    now let's split the train_data
    the performance of the model in an unbiased way
    '''

    # taking features into one dataframe and target into another dataframe

    X = train_data.drop('SalePrice', axis=1)
    y = train_data['SalePrice']

    # let's split the data - 80 % for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # noqa: E501

    print(f"shape after splitting : \n train data {X_train.shape} {y_train.shape} \n test data {X_test.shape} {y_test.shape}")  # noqa: E501
    return X_train, X_test, y_train, y_test


def feature_selection(data):
    # let's check which features are categorical and which are continuous
    print("cont feat :\n", data.select_dtypes(include=['int64']).dtypes.head())
    print("cate:\n", data.select_dtypes(include=['object']).dtypes.head())
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
        if (count != 0):
            print(f"Null values in feature {val} is {count}")
        else:
            print(f"No null value in feature {val}")


def scale_data(data_to_scale, cont_feat):
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
    data_cont_scaled = pd.DataFrame(loaded_scaler.transform(data_cont), columns=data_cont.columns)  # noqa: E501
    print("\nlet's see how scaled data looks like:")
    print(data_cont_scaled.head())
    print("see if range is reduced or not")
    findrange(data_cont_scaled)
    print("shape of data scaled :", data_cont_scaled.shape)
    return data_cont_scaled


def encode_data(data_to_encode, cate_feat, X_train_cate):
    # Categorical features need to be encoded to numerical values.

    # seperate the dataframe  into categorical subset
    data_cate = data_to_encode[cate_feat]

    print(data_cate.head())

    print(f"\nthe shape of data to encode is \n {data_cate.shape}")
    # encoding the categorical values
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    joblib.dump(encoder, '../models/encoder.joblib')
    loaded_encoder = joblib.load('../models/encoder.joblib')
    loaded_encoder.fit(X_train_cate)
    data_cate_encoded = pd.DataFrame(loaded_encoder.transform(data_cate),
        columns=loaded_encoder.get_feature_names_out(data_cate.columns))  # noqa: E128,E501

    print(f"\nthe shape after encoding   \n {data_cate_encoded.shape}")
    print("see if the data is encoded or not")
    print(data_cate_encoded.head())
