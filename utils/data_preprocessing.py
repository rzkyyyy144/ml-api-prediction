import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    data['Month'] = pd.to_datetime(data['Month'])
    data = data.sort_values('Month')

    data = data.tail(1000)

    values = data[['Demand', 'Unit_Price']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    return scaled_data, scaler

def create_sequences(data, seq_length=30):
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        targets.append(data[i, :])
    
    return np.array(sequences), np.array(targets)
