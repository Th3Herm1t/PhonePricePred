from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print("Received data:", data)  # Print the received form data for debugging

    try:
        features = np.array([
            float(data['Ratings']), 
            float(data['Ram']), 
            float(data['ROM']), 
            float(data['Mobile_Size']),
            float(data['Primary_Cam']),
            float(data['Selfi_Cam']),
            float(data['Battery_Power'])
        ])
        scaled_features = scaler.transform([features])
        poly_features = poly.transform(scaled_features)
        prediction = model.predict(poly_features)
        return jsonify({'prediction': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing input: {str(e)}'}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)