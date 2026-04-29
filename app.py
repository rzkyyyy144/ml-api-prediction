from flask import Flask, jsonify # type: ignore
from controller.predict_controller_lstm import predict_next_month, train_model
from controller.predict_controller_items import predict_demand
from utils.data_preprocessing import load_and_preprocess_data

app = Flask(__name__)

data_path = 'data/modified_data_integer.csv'
model_path = 'models/lstm_model.h5'

scaler = train_model(data_path, model_path)

@app.route('/predict', methods=['GET'])
def predict():
    scaled_data, _ = load_and_preprocess_data(data_path)
    recent_data = scaled_data[-30:]

    result = predict_next_month(model_path, scaler, recent_data)
    return jsonify(result)

@app.route('/predict-items', methods=['GET'])
def predict_items():
    result = predict_demand()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

