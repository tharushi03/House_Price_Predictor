from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

# Load model
model_path = 'models/house_price_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("[OK] Model loaded successfully")
else:
    model = None
    print("[WARNING] Model not found. Please run build_model.py first")

@app.route('/')
def home():
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction=None, error="Model not loaded")

    try:
        # Get form data
        overall_qual = int(request.form['overall_qual'])
        overall_cond = int(request.form['overall_cond'])
        year_built = int(request.form['year_built'])
        lot_area = int(request.form['lot_area'])
        gr_liv_area = int(request.form['gr_liv_area'])
        bedrooms = int(request.form['bedrooms'])
        full_bath = int(request.form['full_bath'])
        half_bath = int(request.form['half_bath'])
        fireplaces = int(request.form['fireplaces'])
        kitchen_qual = request.form['kitchen_qual']
        garage_cars = int(request.form['garage_cars'])
        garage_area = int(request.form['garage_area'])
        basement_sf = int(request.form['basement_sf'])
        neighborhood = request.form['neighborhood']

        # Create feature dictionary
        features = {
            'MSSubClass': 60,
            'MSZoning': 'RL',
            'LotFrontage': lot_area / 100,
            'LotArea': lot_area,
            'Street': 'Pave',
            'Alley': None,
            'LotShape': 'Reg',
            'LandContour': 'Lvl',
            'Utilities': 'AllPub',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',
            'Neighborhood': neighborhood,
            'Condition1': 'Norm',
            'Condition2': 'Norm',
            'BldgType': '1Fam',
            'HouseStyle': '2Story',
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            'YearBuilt': year_built,
            'YearRemodAdd': year_built,
            'RoofStyle': 'Gable',
            'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'BrkFace',
            'MasVnrArea': 0,
            'ExterQual': 'Gd',
            'ExterCond': 'TA',
            'Foundation': 'PConc',
            'BsmtQual': 'Gd' if basement_sf > 0 else 'None',
            'BsmtCond': 'TA',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'GLQ',
            'BsmtFinSF1': basement_sf,
            'BsmtFinType2': 'Unf',
            'BsmtFinSF2': 0,
            'BsmtUnfSF': 0,
            'TotalBsmtSF': basement_sf,
            'Heating': 'GasA',
            'HeatingQC': 'Ex',
            'CentralAir': 'Y',
            'Electrical': 'SBrkr',
            '1stFlrSF': gr_liv_area // 2,
            '2ndFlrSF': gr_liv_area // 2,
            'LowQualFinSF': 0,
            'GrLivArea': gr_liv_area,
            'BsmtFullBath': 1 if basement_sf > 700 else 0,
            'BsmtHalfBath': 0,
            'FullBath': full_bath,
            'HalfBath': half_bath,
            'BedroomAbvGr': bedrooms,
            'KitchenAbvGr': 1,
            'KitchenQual': kitchen_qual,
            'TotRmsAbvGrd': 8,
            'Functional': 'Typ',
            'Fireplaces': fireplaces,
            'FireplaceQu': 'Gd' if fireplaces > 0 else None,
            'GarageType': 'Attchd' if garage_cars > 0 else 'None',
            'GarageYrBlt': year_built if garage_cars > 0 else 0,
            'GarageFinish': 'RFn' if garage_cars > 0 else 'None',
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            'GarageQual': 'TA',
            'GarageCond': 'TA',
            'PavedDrive': 'Y',
            'WoodDeckSF': 0,
            'OpenPorchSF': 61,
            'EnclosedPorch': 0,
            '3SsnPorch': 0,
            'ScreenPorch': 0,
            'PoolArea': 0,
            'PoolQC': None,
            'Fence': None,
            'MiscFeature': None,
            'MiscVal': 0,
            'MoSold': 6,
            'YrSold': 2008,
            'SaleType': 'WD',
            'SaleCondition': 'Normal'
        }

        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        prediction = max(50000, min(prediction, 1000000))

        return render_template('index.html', prediction=prediction, error=None)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTOR")
    print("="*50)
    print("\n[OK] Starting server...")
    print("[OK] Open your browser to: http://127.0.0.1:5000")
    print("[INFO] Press CTRL+C to stop the server")
    print("\n" + "="*50)
    app.run(debug=True, host='127.0.0.1', port=5000)
