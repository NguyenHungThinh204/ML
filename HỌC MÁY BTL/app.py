
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the models
models = {
    'linear': joblib.load('HỌC MÁY BTL/linear_regression_model.pkl'),
    'ridge': joblib.load('HỌC MÁY BTL/ridge_regression_model.pkl'),
    'nn': joblib.load('HỌC MÁY BTL/neural_network_model.pkl'),
    'stacking': joblib.load('HỌC MÁY BTL/stacking_neural_ridge_linear_model.pkl')
}


# Route to load the HTML template
@app.route('/')
def demo():
    return render_template('demo.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    algorithm = data['algorithm']

    # Get the corresponding model
    model = models.get(algorithm)
    if not model:
        return jsonify({'error': 'Algorithm not supported'}), 400

    # Extract input features from the data
    try:
        performance = float(data['performance'])
        storage_capacity = float(data['storage_capacity'])
        camera_quality = float(data['camera_quality'])
        battery_life = float(data['battery_life'])
        weight = float(data['weight'])
        age = int(data['age'])
    except ValueError as e:
        return jsonify({'error': 'Invalid input values'}), 400

    # Create a feature array for prediction
    features_list = [performance, storage_capacity, camera_quality, battery_life, weight, age]
    columns = ['Performance', 'Storage capacity', 'Camera quality', 'Battery life', 'Weight', 'age']
    features = pd.DataFrame([features_list], columns=columns)

    # Use the selected model for prediction
    prediction = model.predict(features)


    try:    # Return the result as a JSON response
        return jsonify({
            'prediction': prediction.tolist(),
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000,debug=True)