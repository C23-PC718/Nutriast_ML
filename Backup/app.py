import pandas as pd
# import tensorflowjs as tfjs
# from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = Flask(__name__)

# Load the model
# model = tfjs.converters.load_keras_model("model_tfjs")
loaded_model = tf.keras.models.load_model("model.h5")



# Define cat_cols globally
cat_cols = ["gender", "cholesterol", "gluc"]
num_cols = ["age", "height", "weight"]

# Define the scaler globally
scaler = MinMaxScaler()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])

    # Fit and define the scaler globally
    scaler = MinMaxScaler()
    training_data = pd.read_csv("heart.csv")
    # X_num = scaler.fit_transform(training_data[num_cols])
    
    # Preprocess the data
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = scaler.fit_transform(df[num_cols])
    X_preprocessed = np.concatenate((X_cat, X_num), axis=1)
    
    # Make predictions
    y_pred = loaded_model.predict(X_preprocessed)[0][0]
    
    # Return the response
    if y_pred > 0.5:
        return jsonify({"result": "cardio_present"})
    else:
        return jsonify({"result": "cardio_absent"})

if __name__ == "__main__":
    app.run(debug=False)