# IOT Soil Moisture Prediction Project

A machine learning project for predicting soil moisture levels using environmental data from three regions in Kenya: Kitui, Syokimau, and Turkana.

## 📋 Project Overview

This project develops and compares multiple machine learning models to predict soil moisture content based on:
- **Temperature** (°C)
- **Relative Humidity** (%)
- **Rainfall** (mm)
- **Previous Day Soil Moisture** (lagged feature)

### Objective
Predict current soil moisture (`target_sm`) to support agricultural planning and water resource management in Kenya.

## 📁 Project Structure

```
IOT(PROJECT)/
├── baseline_test.py              # ML model comparison and LSTM training
├── file_merger.py                # Data preparation and feature engineering
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── Kitui_prepared.csv            # Processed Kitui data
├── Syokimau_prepared.csv         # Processed Syokimau data
├── Turkana_prepared.csv          # Processed Turkana data
└── Raw Data Files/               # Original TAMSAT & Open Meteo datasets
    ├── TAMSAT & Open Meteo daily data for 10 yrs for Kitui (soil moisture, rainfall, Av temp& Av humidity).csv
    ├── TAMSAT & Open Meteo daily data for 10 yrs for Syokimau (soil moisture, rainfall, Av temp& Av humidity).csv
    └── TAMSAT & Open Meteo daily data for 10 yrs for Turkana (soil moisture, rainfall, Av temp& Av humidity).csv
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lucasmos/IOT-PROJECT-.git
cd IOT-PROJECT-
```

2. Create a virtual environment:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Pipeline

### 1. Data Preparation (`file_merger.py`)

Processes raw TAMSAT & Open Meteo datasets:
- Creates lagged feature (previous day's soil moisture)
- Converts dates to millisecond timestamps
- Removes missing values
- Renames columns for ML compatibility
- Outputs: `[city]_prepared.csv` files

Run data preparation:
```bash
python file_merger.py
```

### 2. Model Training & Evaluation (`baseline_test.py`)

Trains and evaluates four models:

| Model | Type | Parameters |
|-------|------|-----------|
| **Random Forest** | Ensemble | 100 trees |
| **XGBoost** | Boosting | 100 trees, lr=0.1, depth=3 |
| **LightGBM** | Boosting | 100 trees, lr=0.05 |
| **LSTM** | Deep Learning | 32→16→8→1 neurons, 200 epochs |

Run model training:
```bash
python baseline_test.py
```

## 📈 Features & Input Data

The prepared datasets contain 10 years of daily observations with:

| Feature | Description | Source |
|---------|-------------|--------|
| `timestamp` | Day index in milliseconds | Generated |
| `temp` | Mean daily temperature | Open Meteo |
| `hum` | Mean daily humidity | Open Meteo |
| `rain` | Daily rainfall | TAMSAT |
| `prev_sm` | Previous day soil moisture | TAMSAT (lagged) |
| `target_sm` | Current soil moisture | TAMSAT |

## 🔄 Model Workflow

1. **Data Loading**: Concatenate Kitui and Turkana datasets
2. **Train-Test Split**: 80-20 split with random_state=42
3. **Scaling**: MinMaxScaler normalization (0-1 range)
4. **Model Training**: Parallel training of 4 models
5. **Evaluation**: R2 score, MAE, and RMSE metrics

## 📊 Evaluation Metrics

- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MAE**: Mean Absolute Error in soil moisture units
- **RMSE**: Root Mean Squared Error (penalizes larger errors)

## 🧠 Model Details

### Traditional ML Models
- Trained on scaled features with sklearn preprocessing
- Use dataframes to avoid feature name warnings
- Robust to non-linear relationships

### LSTM Neural Network
- **Architecture**: Sequential model with LSTM cell
  - Input: 4 features (reshaped to timesteps)
  - Layer 1: 32 LSTM units + tanh activation
  - Layer 2: 16 dense neurons + ReLU
  - Layer 3: 8 dense neurons + ReLU
  - Output: 1 neuron (soil moisture prediction)
- **Training**: 200 epochs, Early Stopping (patience=20)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error

## 🚀 Usage

### Run Complete Pipeline
```bash
# Prepare data
python file_merger.py

# Train and evaluate models
python baseline_test.py
```

### Output
The baseline_test.py script prints a comparison table:
```
Model        R2    MAE    RMSE
Random Forest 0.xx  x.xx   x.xx
XGBoost       0.xx  x.xx   x.xx
LightGBM      0.xx  x.xx   x.xx
LSTM          0.xx  x.xx   x.xx
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.0 | ML algorithms & preprocessing |
| xgboost | 2.0.3 | XGBoost regressor |
| lightgbm | 4.1.1 | LightGBM regressor |
| tensorflow | 2.14.0 | Deep learning (LSTM) |
| matplotlib | 3.8.0 | Data visualization |
| seaborn | 0.13.0 | Statistical visualization |

## 🌍 Geographic Coverage

- **Kitui**: Semi-arid region, Central Kenya
- **Syokimau**: Coastal region, Southern Kenya (data available)
- **Turkana**: Arid region, Northern Kenya

## 📝 Data Sources

- **TAMSAT**: Satellite-based soil moisture and rainfall estimates
- **Open Meteo**: Free weather API data (temperature, humidity)

## 🔮 Future Improvements

- [ ] Ensemble predictions combining multiple models
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robust evaluation
- [ ] Time-series specific models (Prophet, ARIMA)
- [ ] Feature importance analysis
- [ ] Model deployment as REST API
- [ ] Web dashboard for real-time predictions
- [ ] Include Syokimau data in final model
- [ ] Uncertainty quantification

## 📄 License

This project is part of an IoT initiative for agricultural applications.

## 👤 Author

**Luke Mosoti**
- Email: lukemosoti19@gmail.com
- GitHub: [lucasmos](https://github.com/lucasmos)

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Last Updated**: February 27, 2026
