def model_analysis(data):
    data.head()
    print(f"\nshape of data {data.shape}")
    print(f"\ndata coloumns : {data.columns}")
    print(f"\nnull values in each features : {data.isnull().sum()}")
    print(f"\ndata types of each features : {data.dtypes}")    
    return


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

