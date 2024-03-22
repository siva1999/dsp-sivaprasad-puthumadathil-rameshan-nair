import streamlit as st
import requests

# Define the API endpoint
API_URL = "http://localhost:8000/predictval/"

# Define the default values for the form
default_values = {
    "GarageArea": 548,
    "HouseStyle": "2Story",
    "Neighborhood": "CollgCr",
    "BldgType": "1Fam",
    "KitchenQual": "Gd",
    "ExterQual": "Gd",
    "OverallQual": 7,
    "YearBuilt": 2003,
    "GrLivArea": 1710,
    "TotalBsmtSF": 856
}

def predict_sale_price(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("PredictedSalePrice")
    else:
        return None

def main():
    st.title("House Price Prediction")

    st.write("Fill in the details below to predict the sale price of a house.")

    # Create a form for user input
    form_values = {}
    for key, value in default_values.items():
        form_values[key] = st.text_input(key, value)

    if st.button("Predict"):
        response = predict_sale_price(form_values)
        if response is not None:
            st.success(f"The predicted sale price is: ${response:.2f}")
        else:
            st.error("Failed to get prediction. Please try again.")

if __name__ == "__main__":
    main()
