import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from utils.data_preprocessing import load_and_preprocess_data, create_sequences
from models.lstm_model import build_lstm_model
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split # type: ignore

def train_model(data_path, model_path, epochs=50, batch_size=32, seq_length=30):
    # Load and scale data
    scaled_data, scaler = load_and_preprocess_data(data_path)

    # Create sequences
    sequences, targets = create_sequences(scaled_data, seq_length)

    # Split data: 80% training, 20% testing
    split_index = int(len(sequences) * 0.8)
    x_train, y_train = sequences[:split_index], targets[:split_index]
    x_test, y_test = sequences[split_index:], targets[split_index:]

    # Build and train model
    model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluate model on test set
    test_loss = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_loss}")

    # Save model
    model.save(model_path)

    return scaler

def predict_next_month(model_path, scaler, recent_data, days_ahead=30):
    model = load_model(model_path)
    
    daily_predictions = []

    for _ in range(days_ahead):
        recent_data_reshaped = recent_data.reshape((1, recent_data.shape[0], recent_data.shape[1]))
        prediction = model.predict(recent_data_reshaped)
        prediction_inversed = scaler.inverse_transform(prediction)
        daily_predictions.append(prediction_inversed[0])
        recent_data = np.append(recent_data, prediction, axis=0)
        recent_data = recent_data[1:]

    daily_predictions = np.array(daily_predictions)

    predicted_total_demand = int(daily_predictions[:, 0].sum())
    predicted_total_unit_price = int(daily_predictions[:, 1].sum())
    
    avg_daily_demand = int(predicted_total_demand / days_ahead)
    historical_avg = np.mean(scaler.inverse_transform(recent_data)[:, 0])
    demand_percentage = float((predicted_total_demand / historical_avg) * 100)

    if demand_percentage >= 120:
        demand_category = "HIGH"
    elif demand_percentage >= 80:
        demand_category = "NORMAL"
    else:
        demand_category = "LOW"

    predicted_month = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m")
    prediction_date = datetime.now().strftime("%Y-%m-%d")

    return {
    "predicted_avg_unit_price": predicted_total_unit_price,
    "predicted_demand": predicted_total_demand,
    "avg_daily_demand": avg_daily_demand,
    "demand_percentage": round(demand_percentage, 2),
    "demand_category": demand_category,
    "predicted_month": predicted_month,
    "prediction_date": prediction_date
}
