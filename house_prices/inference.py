import pandas as pd
import numpy as np
import joblib
from app import cate_feat,cont_feat

def imputation(test_data):
    test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].median(), inplace=True)  # noqa: E501
    test_data['GarageArea'].fillna(test_data['GarageArea'].median(), inplace=True)  # noqa: E501
    test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0], inplace=True)  # noqa: E501


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:

    imputation(input_data)
    test_cont = input_data[cont_feat]
    test_cate = input_data[cate_feat]
    loaded_scaler = joblib.load('../models/scaler.joblib')

    test_cont_scaled = pd.DataFrame(loaded_scaler.transform(test_cont), columns=test_cont.columns)  # noqa: E501

    loaded_encoder = joblib.load('../models/encoder.joblib')

    test_cate_encoded = pd.DataFrame(loaded_encoder.transform(test_cate),
                                     columns=loaded_encoder.get_feature_names_out(test_cate.columns))

    test_preprocessed = pd.concat([test_cont_scaled, test_cate_encoded], axis=1)  # noqa: E501
    
    loaded_model = joblib.load('../models/model.joblib')
    test_pred = loaded_model.predict(test_preprocessed)

    print("The first 10 predicted SalePrice values :\n", test_pred[:10])

