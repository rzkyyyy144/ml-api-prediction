import xgboost as xgb # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore

def train_xgb_model(df):
    feature_cols = ['Price', 'DayOfWeek', 'Month', 'AvgDemand', 'RequestFrequency']
    target_col = 'Total'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"XGBoost MAE: {mae}")

    return model
