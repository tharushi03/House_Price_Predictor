"""
build_model.py - Train and save the house price prediction model
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("="*60)
print("HOUSE PRICE PREDICTION MODEL - TRAINING")
print("="*60)

# Load data
data_path = "data/train.csv"
print(f"\n1. Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"   Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Define target
TARGET = 'SalePrice'
print(f"\n2. Target column: {TARGET}")

# Prepare features
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Remove ID column
if 'Id' in X.columns:
    X = X.drop('Id', axis=1)
    print("   Removed 'Id' column")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n3. Data split:")
print(f"   Training: {X_train.shape[0]} rows")
print(f"   Test: {X_test.shape[0]} rows")

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n4. Feature types:")
print(f"   Numeric: {len(numeric_cols)}")
print(f"   Categorical: {len(categorical_cols)}")

# Create preprocessing pipelines
print(f"\n5. Building preprocessing pipeline...")
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train the model
print(f"\n6. Training model...")
model.fit(X_train, y_train)
print("   Training complete!")

# Evaluate
print(f"\n7. Evaluating model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")

# Save the model
print(f"\n8. Saving model...")
os.makedirs('models', exist_ok=True)
model_path = 'models/house_price_model.pkl'
joblib.dump(model, model_path)
print(f"   Model saved to: {model_path}")

# Save feature info
feature_info = {
    'numeric_features': numeric_cols,
    'categorical_features': categorical_cols,
    'target': TARGET,
    'performance': {
        'rmse': float(rmse),
        'r2': float(r2)
    }
}
with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"   Feature info saved")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
