import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Load your prepared data
# You can merge your Kitui and Turkana files for a robust training set
df = pd.read_csv("Kitui_prepared.csv") 
df_turkana = pd.read_csv("Turkana_prepared.csv")

X = df[['temp', 'hum', 'rain', 'prev_sm']]
y = df['target_sm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Baseline Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
}

# 3. Train and Get Comparison Data
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)})

# 4. Display results for your PowerPoint
comparison_df = pd.DataFrame(results)
print("\n--- Baseline Model Comparison Table ---")
print(comparison_df.to_string(index=False))