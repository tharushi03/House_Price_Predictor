# House_Price_Predictor
AI-Powered Real Estate Price Estimation
<img width="1097" height="735" alt="image" src="https://github.com/user-attachments/assets/18d2dd7f-6ffb-4713-8f8f-35bb72bc8c0e" />
 <img width="1126" height="802" alt="image" src="https://github.com/user-attachments/assets/859f495c-81c4-4f43-8224-47d77551ae23" />
 <img width="1112" height="768" alt="image" src="https://github.com/user-attachments/assets/32ed2e2e-32bc-4e72-916b-e35234fedd65" />


Step 1: Install Required Packages
Open Command Prompt (CMD) or Terminal and run:

bash
pip install -r requirements.txt
Or install manually:

bash
pip install flask pandas numpy scikit-learn joblib

Step 2: Train the Model
bash
python build_model.py

This will:

	Load the training data
	Preprocess features (handle missing values, encode categories)
	Train a Random Forest model
	Save the model to models/house_price_model.pkl

Step 3: Run the Web Application

bash
python app.py

Step 4: Open in Browser
Once the server starts, open your browser and go to:

text
http://127.0.0.1:5000
