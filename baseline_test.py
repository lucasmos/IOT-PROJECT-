import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
model_predictions = {}
for name, model in models.items():
    model.fit(X_train_df, y_train)
    preds = model.predict(X_test_df)
    model_predictions[name] = preds

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

# 10. Visualizations
sns.set_style("whitegrid")
sns.set_palette("husl")

# 10.1 Model Comparison - Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')

# R2 Score comparison
axes[0].bar(comparison_df['Model'], comparison_df['R2'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
axes[0].set_title('R² Score (Higher is Better)', fontsize=12)
axes[0].set_ylim([0, 1])
for i, v in enumerate(comparison_df['R2']):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# MAE comparison
axes[1].bar(comparison_df['Model'], comparison_df['MAE'], color='coral', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
axes[1].set_title('Mean Absolute Error (Lower is Better)', fontsize=12)
for i, v in enumerate(comparison_df['MAE']):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

# RMSE comparison
axes[2].bar(comparison_df['Model'], comparison_df['RMSE'], color='lightgreen', alpha=0.7, edgecolor='black')
axes[2].set_ylabel('RMSE', fontsize=11, fontweight='bold')
axes[2].set_title('Root Mean Squared Error (Lower is Better)', fontsize=12)
for i, v in enumerate(comparison_df['RMSE']):
    axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison.png")
plt.show()

# 10.2 Actual vs Predicted for each model
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Actual vs Predicted Soil Moisture", fontsize=16, fontweight='bold')

model_names = list(model_predictions.keys()) + ["LSTM"]
predictions_list = list(model_predictions.values()) + [lstm_preds]

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    ax.scatter(y_test, preds, alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_title(f'{name} (R² = {comparison_df.loc[comparison_df["Model"] == name, "R2"].values[0]:.4f})', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: actual_vs_predicted.png")
plt.show()

# 10.3 Residuals Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Residuals Analysis", fontsize=16, fontweight='bold')

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    residuals = y_test - preds
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
print("✓ Saved: residuals_analysis.png")
plt.show()

# 10.4 Distribution of Predictions vs Actual
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribution: Actual vs Predicted", fontsize=16, fontweight='bold')

for idx, (name, preds) in enumerate(zip(model_names, predictions_list)):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    ax.hist(y_test, bins=30, alpha=0.6, label='Actual', color='steelblue', edgecolor='black')
    ax.hist(preds, bins=30, alpha=0.6, label='Predicted', color='coral', edgecolor='black')
    
    ax.set_xlabel('Soil Moisture', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: distribution_comparison.png")
plt.show()

# 10.5 Summary statistics
print("\n--- Visualization Summary ---")
print("Generated 4 visualization files:")
print("  1. model_comparison.png - Bar charts of R², MAE, RMSE")
print("  2. actual_vs_predicted.png - Scatter plots for each model")
print("  3. residuals_analysis.png - Residual distributions")
print("  4. distribution_comparison.png - Histogram comparisons")
print("\nAll plots saved in the project directory.")