from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the saved KMeans model and scaler
kmeans = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a route to display a simple HTML page
@app.route('/')
def index():
    return render_template('index.html')  # This will render an HTML form

# Define a route for prediction (this will be called via POST request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON data
        data = request.form.to_dict()
        print(data)

        # Convert data to the expected input format (list of features)
        input_features = [
            'Age', 'Gender', 'Location', 'AvgTxnAmount',
            'TotalTxnAmount', 'TxnCount', 'AvgAccountBalance', 'TxnFrequency'
        ]

        # Extract the features from the received data
        features = [float(data[feature]) for feature in input_features]

        # Reshape and scale the features
        features_scaled = scaler.transform([features])

        # Make prediction using the KMeans model
        cluster = kmeans.predict(features_scaled)

        # Return the cluster prediction as a response
        return jsonify({'cluster': int(cluster[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
