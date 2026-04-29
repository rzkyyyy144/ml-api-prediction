import pandas as pd # type: ignore
from utils.data_preprocessing_items import preprocess_data
from models.regression_model_items import train_xgb_model
import numpy as np # type: ignore

def predict_demand():
    file_path = 'data/training-data.csv'
    df, scaler_price = preprocess_data(file_path)

    today = pd.Timestamp.today()
    next_month = (today + pd.DateOffset(months=1)).strftime('%Y-%m')

    model = train_xgb_model(df)

    df['Predicted_Demand'] = model.predict(df[['Price', 'DayOfWeek', 'Month', 'AvgDemand', 'RequestFrequency']])
    df['Predicted_Demand'] = np.round(df['Predicted_Demand']).astype(int)
    df['Predicted_Demand'] = np.maximum(df['Predicted_Demand'], 0)

    df_sorted = df.sort_values(by='Predicted_Demand', ascending=False).drop_duplicates(subset=['Item'], keep='first')

    df_sorted['Price'] = df_sorted['Original_Price'].apply(format_rupiah)
    df_filtered = df_sorted[df_sorted['Predicted_Demand'] >= 1]

    top_50 = df_filtered.head(50)[['Item', 'Satuan', 'Price', 'Predicted_Demand', 'RequestFrequency']].to_dict(orient='records')
    low_50 = df_filtered.tail(50)[['Item', 'Satuan', 'Price', 'Predicted_Demand', 'RequestFrequency']].to_dict(orient='records')

    return {
        "predicted_month": next_month,
        "prediction_date": today.strftime('%Y-%m-%d'),
        "top_50": top_50,
        "low_50": low_50
    }

def format_rupiah(price):
    return f"{int(price):,}".replace(",", ".")
