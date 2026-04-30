import pandas as pd

# 1. Load the local Syokimau data
df_syokimau = pd.read_csv('syokimau_data_fully_cleaned_v2.csv')

# 2. Create the 'target_sm' column by shifting the 'prev_sm' column UP by 1 row
df_syokimau['target_sm'] = df_syokimau['prev_sm'].shift(-1)

# 3. Drop the very last row because it won't have a future 'target_sm' to predict
df_syokimau = df_syokimau.dropna(subset=['target_sm'])

# 4. Ensure the column order exactly matches the Kitui training data
expected_columns = ['timestamp', 'temp', 'hum', 'rain', 'prev_sm', 'target_sm']
df_syokimau = df_syokimau[expected_columns]

# 5. Save the prepared dataset
df_syokimau.to_csv('Syokimau_local_prepared.csv', index=False)

print("Data formatting complete! Saved as 'Syokimau_prepared.csv'.")