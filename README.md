AgriPredict AI

AgriPredict AI is a machine learning-based web application that predicts:
Crop Yield (Quintals per Hectare)
Agricultural Market Price (Modal Price per Quintal)

The system uses Random Forest Regressor and is deployed using Flask.

Features
Crop Yield Prediction
Market Price Prediction
Web-based user interface
Model evaluation using MAE, RMSE, and R² score

Technologies Used
Python
Scikit-learn
Pandas
NumPy
Flask
Matplotlib


How to Run the Project

Install dependencies:
pip install -r requirements.txt

Train models:
python crop_yield.py
python market_price.py

Run the application:
python app.py


Open in browser:
http://127.0.0.1:5000 -- with the link in the terminal after running app.py



Model Information
Algorithm Used: Random Forest Regressor
Evaluation Metrics: MAE, RMSE, R² Score

Target Variables:
Crop Yield (Quintals per Hectare)
Modal Market Price (₹ per Quintal)