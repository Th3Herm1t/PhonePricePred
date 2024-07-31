import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the saved model
model = joblib.load('best_model_Random Forest.pkl')  # Adjust the file name if necessary

# Initialize Flask app
app = Flask(__name__)

# Initialize the scaler and polynomial features (ensure these match your training pipeline)
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)

# Dummy data for fitting the scaler and poly (replace with your training data if available)
dummy_data = np.array([[4, 64, 5.5, 12, 8, 4000]])
scaler.fit(dummy_data)
poly.fit(scaler.transform(dummy_data))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form
    features = np.array([
        float(data['Ram']),
        float(data['ROM']),
        float(data['Mobile_Size']),
        float(data['Primary_Cam']),
        float(data['Selfi_Cam']),
        float(data['Battery_Power'])
    ]).reshape(1, -1)
    
    # Transform the input features
    scaled_features = scaler.transform(features)
    poly_features = poly.transform(scaled_features)
    
    # Make prediction
    prediction = model.predict(poly_features)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
