import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from sqlalchemy import create_engine, Column, Integer,Float, Boolean, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
from app import cate_feat, cont_feat

from house_prices.inference import imputation, make_predictions

sys.path.append("..")

app = FastAPI()

DATABASE_URL = "postgresql://siva:siva@localhost/jsp"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FeatureInput(Base):
    __tablename__ = "houseprice"
    id = Column(Integer, primary_key=True, index=True)
    GarageArea = Column(Integer)
    HouseStyle = Column(String)
    Neighborhood = Column(String)
    BldgType = Column(String)
    KitchenQual = Column(String)
    ExterQual = Column(String)
    OverallQual = Column(Integer)
    YearBuilt = Column(Integer)
    GrLivArea = Column(Integer)
    TotalBsmtSF = Column(Integer)
    PredictedSalePrice = Column(Float)


class FeatureInputRequest(BaseModel):
    GarageArea: int
    HouseStyle: str
    Neighborhood: str
    BldgType: str
    KitchenQual: str
    ExterQual: str
    OverallQual: int
    YearBuilt: int
    GrLivArea: int
    TotalBsmtSF: int


# API endpoint to receive input features, store them in the database, make predictions, and return the result
@app.post("/predictval/")
async def predict_features(features: FeatureInputRequest):
    db = SessionLocal()
    try:
        # Store input features in the database
        feature_input = FeatureInput(**features.dict())
        db.add(feature_input)
        db.commit()
        db.refresh(feature_input)
  
        input_data = pd.DataFrame(features.dict(), index=[0])  # Pass index=[0] to indicate one row of data.
        imputation(input_data)
        test_cont = input_data[cont_feat]
        test_cate = input_data[cate_feat]
        loaded_scaler = joblib.load('models/scaler.joblib')

        test_cont_scaled = pd.DataFrame(loaded_scaler.transform(test_cont), columns=test_cont.columns)  # noqa: E501

        loaded_encoder = joblib.load('models/encoder.joblib')

        test_cate_encoded = pd.DataFrame(loaded_encoder.transform(test_cate),
                                        columns=loaded_encoder.get_feature_names_out(test_cate.columns))  # noqa: E501

        test_preprocessed = pd.concat([test_cont_scaled, test_cate_encoded], axis=1)  # noqa: E501
        loaded_model = joblib.load('models/model.joblib')
        test_pred = loaded_model.predict(test_preprocessed)

        feature_input.PredictedSalePrice = test_pred[0] 
        db.add(feature_input)
        db.commit()
        db.refresh(feature_input)

        return JSONResponse(content={"PredictedSalePrice": float(test_pred[0])})
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail= f"Internal Server Error {str(e)}")
    finally:
        db.close()
