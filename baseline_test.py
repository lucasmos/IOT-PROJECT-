import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

Sequential = tf.keras.models.Sequential
InputLayer = tf.keras.layers.InputLayer
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping

# 1. Load Data Explicitly into Train, Test, and Validation Roles
print("Loading datasets...")
df_kitui = pd.read_csv("Kitui_prepared.csv")
df_turkana = pd.read_csv("Turkana_prepared.csv")
df_syokimau_hist = pd.read_csv("Syokimau_prepared.csv")
df_syokimau_local = pd.read_csv("Syokimau_local_prepared.csv")

# Enforce the requested splits without shuffling time-series data
df_train = pd.concat([df_kitui, df_turkana], ignore_index=True)
df_test = df_syokimau_hist
df_val = df_syokimau_local

features = ['temp', 'hum', 'rain', 'prev_sm']

X_train = df_train[features].values
y_train = df_train['target_sm'].values

X_test = df_test[features].values
y_test = df_test['target_sm'].values

X_val = df_val[features].values
y_val = df_val['target_sm'].values

# Print the split distribution to verify the conceptual 70/20/10 distribution
total_samples = len(df_train) + len(df_test) + len(df_val)
print(f"\n--- Data Split Breakdown ---")
print(f"Training (Kitui + Turkana): {len(df_train)} samples ({len(df_train)/total_samples*100:.1f}%)")
print(f"Testing (Syokimau Historical): {len(df_test)} samples ({len(df_test)/total_samples*100:.1f}%)")
print(f"Validation (Syokimau Local Hardware): {len(df_val)} samples ({len(df_val)/total_samples*100:.1f}%)\n")

# 2. Scale data (Fit strictly on training data to prevent future data leakage)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

# 3. Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, 4))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, 4))
X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, 4))

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
# We use the Testing set (Historical Syokimau) for Early Stopping, leaving the Local data 100% unseen
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("\nTraining LSTM Model...")
lstm_model.fit(
    X_train_lstm, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_data=(X_test_lstm, y_test_scaled),
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate LSTM strictly on the Local Validation Set
lstm_preds_scaled = lstm_model.predict(X_val_lstm)
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()

lstm_r2 = r2_score(y_val, lstm_preds)
lstm_mae = mean_absolute_error(y_val, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_val, lstm_preds))

# 7. Train & Evaluate Baseline Models strictly on Validation Data
X_train_df = pd.DataFrame(X_train, columns=features)
X_val_df = pd.DataFrame(X_val, columns=features)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
}

results = []
model_predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_df, y_train)
    preds = model.predict(X_val_df)
    model_predictions[name] = preds

    results.append({
        "Model": name,
        "R2": round(r2_score(y_val, preds), 4),
        "MAE": round(mean_absolute_error(y_val, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_val, preds)), 4)
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
print("\n--- HARDWARE VALIDATION RESULTS (Syokimau Local Data) ---")
print(comparison_df.to_string(index=False))

# 10. Visualizations (Evaluating how the models performed on the Physical Deployment)
sns.set_style("whitegrid")
sns.set_palette("husl")

# 10.1 Model Comparison - Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Local Validation Performance Comparison", fontsize=16, fontweight='bold')

axes[0].bar(comparison_df['Model'], comparison_df['R2'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[0].set_title('R² Score (Higher is Better)', fontsize=12)
axes[0].set_ylim([0, 1])
for i, v in enumerate(comparison_df['R2']):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(comparison_df['Model'], comparison_df['MAE'], color='coral', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
axes[1].set_title('Mean Absolute Error (Lower is Better)', fontsize=12)
for i, v in enumerate(comparison_df['MAE']):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(comparison_df['Model'], comparison_df['RMSE'], color='lightgreen', alpha=0.7, edgecolor='black')
axes[2].set_ylabel('RMSE', fontsize=11, fontweight='bold')
axes[2].set_title('Root Mean Squared Error (Lower is Better)', fontsize=12)
for i, v in enumerate(comparison_df['RMSE']):
    axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.2 Actual vs Predicted (Local Data)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Actual vs Predicted Soil Moisture (Syokimau Local)", fontsize=16, fontweight='bold')

model_names = list(model_predictions.keys()) + ["LSTM"]
predictions_list = list(model_predictions.values()) + [lstm_preds]

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    ax.scatter(y_val, preds, alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    
    min_val = min(y_val.min(), preds.min())
    max_val = max(y_val.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Local Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_title(f'{name} (R² = {comparison_df.loc[comparison_df["Model"] == name, "R2"].values[0]:.4f})', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.3 Residuals Analysis (Local Data)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Residuals Analysis on Local Hardware Data", fontsize=16, fontweight='bold')

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    residuals = y_val - preds
    ax.scatter(preds, residuals, alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    
    ax.set_xlabel('Predicted Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=10, fontweight='bold')
    ax.set_title(f'{name} Residuals (RMSE = {comparison_df.loc[comparison_df["Model"] == name, "RMSE"].values[0]:.4f})',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.4 Distribution of Predictions vs Actual
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribution: Actual vs Predicted (Syokimau Local)", fontsize=16, fontweight='bold')

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    ax.hist(y_val, bins=30, alpha=0.6, label='Actual Local', color='steelblue', edgecolor='black')
    ax.hist(preds, bins=30, alpha=0.6, label='Predicted', color='coral', edgecolor='black')
    
    ax.set_xlabel('Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n--- Visualization Summary ---")
print("Generated 4 visualization files based strictly on the Local Validation Set:")
print("  1. model_comparison.png")
print("  2. actual_vs_predicted.png")
print("  3. residuals_analysis.png")
print("  4. distribution_comparison.png")
print("\nValidation complete. All plots saved to the repository.")