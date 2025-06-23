# using post and postman
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained SVM model
model_path = os.path.join("..", "svm_iris_model.pkl")
model = joblib.load(model_path)

@app.route("/Yin")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        data = request.get_json()

        # Extract features from JSON
        features = [[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]]

        # Convert to DataFrame for feature names (optional to avoid warnings)
        df = pd.DataFrame(features, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])

        # Make prediction
        prediction = model.predict(df)[0]

        # Respond with input data and prediction
        return jsonify({
            "message": "Prediction successful",
            "input_data": data,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)