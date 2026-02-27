import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf

Sequential = tf.keras.models.Sequential
InputLayer = tf.keras.layers.InputLayer
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping

# 1. Load Data
df = pd.read_csv("Kitui_prepared.csv")
df_turkana = pd.read_csv("Turkana_prepared.csv")
df = pd.concat([df, df_turkana], ignore_index=True)

features = ['temp', 'hum', 'rain', 'prev_sm']
X = df[features].values
y = df['target_sm'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# 3. Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, 4))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, 4))

# 4. Build LSTM Model
lstm_model = Sequential([
    InputLayer(input_shape=(1, 4)),
    LSTM(32, activation='tanh', recurrent_activation='sigmoid'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, name='y_pred')
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
lstm_model.summary()

# 5. Train LSTM
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

lstm_model.fit(
    X_train_lstm, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate LSTM
lstm_preds_scaled = lstm_model.predict(X_test_lstm)
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()

lstm_r2 = r2_score(y_test, lstm_preds)
lstm_mae = mean_absolute_error(y_test, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))

# 7. Baseline Models — pass X_test as DataFrame to avoid feature name warning
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df = pd.DataFrame(X_test, columns=features)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
}

results = []
for name, model in models.items():
    model.fit(X_train_df, y_train)
    preds = model.predict(X_test_df)

    results.append({
        "Model": name,
        "R2": round(r2_score(y_test, preds), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4)
    })

# 8. Add LSTM to results
results.append({
    "Model": "LSTM",
    "R2": round(lstm_r2, 4),
    "MAE": round(lstm_mae, 4),
    "RMSE": round(lstm_rmse, 4)
})

# 9. Print comparison table
comparison_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))