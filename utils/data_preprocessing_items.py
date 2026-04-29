import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    demand_stats = df.groupby('Item')['Total'].agg(['mean', 'count']).reset_index()
    demand_stats.rename(columns={'mean': 'AvgDemand', 'count': 'RequestFrequency'}, inplace=True)
    
    df = df.merge(demand_stats, on='Item', how='left')

    # **Simpan harga asli sebelum normalisasi**
    df['Original_Price'] = df['Price'] 

    scaler_price = MinMaxScaler()
    df[['Price']] = scaler_price.fit_transform(df[['Price']])

    return df, scaler_price
