# IoT Precision Irrigation System

An end-to-end precision agriculture system combining ESP32 firmware, LoRa wireless communication, edge AI inference, and machine learning to automate irrigation decisions for smallholder farms in Kenya.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Firmware Components](#firmware-components)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Datasets](#datasets)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Visualization Outputs](#visualization-outputs)
- [Dependencies](#dependencies)
- [Geographic Coverage](#geographic-coverage)
- [Data Sources](#data-sources)
- [Author](#author)

---

## Project Overview

This IoT project addresses water scarcity in Kenyan agriculture by automating irrigation using AI-powered soil moisture prediction. The system:

- Reads temperature, humidity, rainfall, and soil moisture data from sensors on an ESP32 field node
- Runs an Edge Impulse neural network **on-device** to predict future soil moisture
- Activates a water pump only when both current and predicted moisture fall below 35%
- Dispenses a precise target volume (150 ml per cycle) measured by a YF-S201 flow sensor
- Logs all sensor readings to an SD card and syncs to a cloud dashboard via a LoRa gateway
- Operates on a 2-minute deep-sleep cycle for power efficiency

The ML pipeline trains and compares four models (Random Forest, XGBoost, LightGBM, LSTM) on 10 years of satellite and weather data from three Kenyan climatic zones to select the best predictor for deployment via Edge Impulse.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FIELD NODE (ESP32)                     │
│                                                          │
│  Sensors                                                 │
│  ├── DHT22          → Temperature + Humidity             │
│  ├── Capacitive     → Soil Moisture (analog pin 39)      │
│  └── YF-S201        → Water Flow (interrupt pin 4)       │
│                                                          │
│  Processing                                              │
│  ├── Edge Impulse AI → Soil moisture prediction          │
│  └── Decision Logic → moisture < 35% AND predicted < 35% │
│                                                          │
│  Actuators & Storage                                     │
│  ├── Relay Pump     → Water pump (active LOW, pin 5)     │
│  ├── SD Card (HSPI) → /syokimau_data.csv + /backlog.csv  │
│  └── I2C LCD 16×2   → Live readings display              │
└──────────────────────┬──────────────────────────────────┘
                       │ LoRa 433 MHz
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  GATEWAY NODE (ESP32)                    │
│                                                          │
│  ├── LoRa receiver  → Parses field node packets          │
│  ├── Open Meteo API → Live rainfall data                 │
│  ├── Blynk Cloud    → V0–V7 virtual pins dashboard       │
│  └── Watchdog timer → 15-second timeout                  │
└──────────────────────┬──────────────────────────────────┘
                       │ WiFi
                       ▼
              ┌─────────────────┐
              │  Blynk Dashboard │
              │  Real-time cloud │
              │  monitoring      │
              └─────────────────┘
```

---

## Repository Structure

```
IOT(PROJECT)/
│
├── Firmware (C++ — ESP32)
│   ├── Field_Node_Code.cpp                  # Master field node with AI & pump control
│   ├── Gateway_Node_Code.cpp                # LoRa–WiFi–Blynk gateway bridge
│   ├── YF-S201_tester.cpp                   # Water flow sensor calibration tool
│   └── soil_moisture_sensor_callibration.cpp # Capacitive sensor calibration tool
│
├── Machine Learning Pipeline (Python)
│   ├── baseline_test.py                     # Multi-model training & evaluation
│   ├── file_merger.py                       # Data preparation & feature engineering
│   ├── local_data_cleaner.py               # Hardware deployment data formatter
│   └── requirements.txt                     # Python dependencies
│
├── Datasets (CSV — 10 years of daily data)
│   ├── TAMSAT & Open Meteo daily data for 10 yrs for Kitui ...csv
│   ├── TAMSAT & Open Meteo daily data for 10 yrs for Syokimau ...csv
│   ├── TAMSAT & Open Meteo daily data for 10 yrs for Turkana ...csv
│   ├── Kitui_prepared.csv                   # Processed Kitui dataset
│   ├── Syokimau_prepared.csv                # Processed Syokimau dataset
│   ├── Turkana_prepared.csv                 # Processed Turkana dataset
│   ├── Syokimau_local_prepared.csv          # Local hardware validation data
│   └── syokimau_data_fully_cleaned_v2.csv   # Cleaned hardware deployment data
│
├── Visualization Outputs (PNG)
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residuals_analysis.png
│   └── distribution_comparison.png
│
├── .vscode/
│   ├── c_cpp_properties.json
│   └── settings.json
├── .gitignore
└── README.md
```

---

## Firmware Components

### Field_Node_Code.cpp — Master Irrigation Controller

The primary embedded application running on the field-deployed ESP32.

**SPI Pin Configuration**

| Bus  | Purpose  | SCK | MISO | MOSI | CS  | RST | DIO0 |
|------|----------|-----|------|------|-----|-----|------|
| VSPI | LoRa     | 18  | 19   | 23   | 15  | 32  | 26   |
| HSPI | SD Card  | 14  | 25   | 13   | 27  | —   | —    |

**Calibration Constants**

| Parameter | Value |
|-----------|-------|
| Dry voltage (air) | 2.42 V |
| Wet voltage (water) | 1.71 V |
| Irrigation threshold | 35% moisture |
| Target volume per cycle | 150 ml |
| Flow calibration factor | 7.5 pulses/sec per L/min |

**Operational Loop (every 2 minutes)**

1. Wake from deep sleep
2. Read DHT22 (temperature + humidity)
3. Read capacitive soil moisture sensor (64-sample average)
4. Request rainfall from gateway via LoRa
5. Run Edge Impulse inference on `[temp, hum, rain, prev_sm]`
6. Irrigate if `current_sm < 35%` AND `predicted_sm < 35%`
7. Count flow pulses until 150 ml dispensed; deactivate pump
8. Log row to `/syokimau_data.csv`; queue offline to `/backlog.csv` if gateway unavailable
9. Sync backlog when gateway comes back online
10. Return to deep sleep

---

### Gateway_Node_Code.cpp — LoRa Gateway Bridge

Bridges field nodes to the cloud over WiFi.

- Receives LoRa packets and parses sensor payloads from field nodes
- Fetches live rainfall data from the Open Meteo API
- Publishes to Blynk virtual pins V0–V7 for dashboard monitoring
- Forwards historical backlog records when a field node reconnects
- Protected by a 15-second hardware watchdog timer

---

### YF-S201_tester.cpp — Flow Sensor Calibration

Standalone sketch to verify and calibrate the YF-S201 water flow sensor. Uses a FALLING-edge interrupt on the pulse pin and the standard calibration factor of 7.5 pulses/second per L/min.

---

### soil_moisture_sensor_callibration.cpp — Capacitive Sensor Calibration

Interactive two-point calibration tool:

1. Prompts user to hold sensor in air → records dry ADC average
2. Prompts user to submerge sensor → records wet ADC average
3. Computes `air_voltage` and `water_voltage` calibration constants
4. Displays live percentage readings with ASCII bar graph

Uses 64-sample ADC averaging for stable readings.

---

## Machine Learning Pipeline

### file_merger.py — Data Preparation

Transforms raw TAMSAT + Open Meteo CSVs into ML-ready feature sets:

| Step | Action |
|------|--------|
| 1 | Create `prev_sm` (lagged previous-day soil moisture) |
| 2 | Convert dates to millisecond Unix timestamps |
| 3 | Drop rows with NaN values |
| 4 | Rename columns to `[timestamp, temp, hum, rain, prev_sm, target_sm]` |
| 5 | Write `[city]_prepared.csv` |

```bash
python file_merger.py
```

---

### local_data_cleaner.py — Hardware Data Formatter

Prepares locally-collected Syokimau hardware deployment data for validation:
- Creates `target_sm` by shifting `prev_sm` forward one day
- Aligns column names with training data schema
- Outputs `Syokimau_local_prepared.csv`

```bash
python local_data_cleaner.py
```

---

### baseline_test.py — Model Training & Evaluation

Trains and compares four models using a geographic train/test/validation split.

**Data Split**

| Split | Dataset | Proportion |
|-------|---------|-----------|
| Training | Kitui + Turkana | 70% |
| Test | Syokimau historical | 20% |
| Validation | Syokimau local hardware | 10% |

**Models**

| Model | Parameters |
|-------|-----------|
| Random Forest | 100 trees |
| XGBoost | 100 trees, lr=0.1, max_depth=3 |
| LightGBM | 100 trees, lr=0.05 |
| LSTM | 32→16→8→1 neurons, 200 epochs, early stopping (patience=20) |

**LSTM Architecture**

```
Input (4 features, reshaped to timesteps)
  └── LSTM (32 units, tanh activation)
       └── Dense (16 units, ReLU)
            └── Dense (8 units, ReLU)
                 └── Dense (1 unit) → predicted soil moisture %
Optimizer: Adam (lr=0.001) | Loss: MSE | Scaling: MinMaxScaler
```

**Evaluation Metrics**

- **R²** — coefficient of determination (higher is better)
- **MAE** — mean absolute error in soil moisture percentage points
- **RMSE** — root mean squared error (penalises large errors more heavily)

```bash
python baseline_test.py
```

---

## Datasets

10 years of daily observations (TAMSAT satellite + Open Meteo weather API) for three Kenyan climatic zones.

**Feature Schema**

| Column | Type | Units | Source |
|--------|------|-------|--------|
| `timestamp` | uint32 | ms | Generated |
| `temp` | float | °C | Open Meteo |
| `hum` | float | % | Open Meteo |
| `rain` | float | mm | TAMSAT |
| `prev_sm` | float | % | TAMSAT (lagged) |
| `target_sm` | float | % | TAMSAT |

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Microcontroller | ESP32 (×2: field node + gateway) |
| Soil moisture sensor | Capacitive (3.3 V, analog output) |
| Temperature/humidity | DHT22 |
| Water flow | YF-S201 |
| Wireless | LoRa SX1278 433 MHz module |
| Display | I2C LCD 16×2 (address 0x27) |
| Storage | MicroSD card module (SPI) |
| Actuator | 5 V relay module + water pump |
| AI | Edge Impulse model deployed as C++ library |

---

## Software Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | ML pipeline |
| Arduino IDE or PlatformIO | Latest | Firmware compilation |
| Edge Impulse CLI | Latest | Model deployment |
| Blynk app | Latest | Cloud dashboard |

---

## Setup & Installation

### Python ML Pipeline

```bash
# Clone the repository
git clone https://github.com/lucasmos/IOT-PROJECT-.git
cd "IOT-PROJECT-"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Firmware

1. Open `Field_Node_Code.cpp` or `Gateway_Node_Code.cpp` in Arduino IDE or PlatformIO.
2. Install required libraries:
   - `LoRa` (Sandeep Mistry)
   - `DHT sensor library` (Adafruit)
   - `LiquidCrystal_I2C`
   - `SD`
   - `BlynkSimpleEsp32`
   - `Edge Impulse SDK` (exported from your Edge Impulse project)
3. Set your credentials in `Gateway_Node_Code.cpp`:
   ```cpp
   const char* ssid     = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   char auth[]          = "YOUR_BLYNK_TOKEN";
   ```
4. Flash `Field_Node_Code.cpp` to the field node ESP32.
5. Flash `Gateway_Node_Code.cpp` to the gateway ESP32.

---

## Usage

### Run the Full ML Pipeline

```bash
# Step 1 — prepare raw data
python file_merger.py

# Step 2 — format hardware validation data
python local_data_cleaner.py

# Step 3 — train and evaluate all models
python baseline_test.py
```

### Sensor Calibration (one-time setup)

Flash `soil_moisture_sensor_callibration.cpp` to the field node, follow the serial prompts, then record the `air_voltage` and `water_voltage` constants into `Field_Node_Code.cpp`.

### Expected Output

```
Model          R2      MAE     RMSE
Random Forest  0.xx    x.xx    x.xx
XGBoost        0.xx    x.xx    x.xx
LightGBM       0.xx    x.xx    x.xx
LSTM           0.xx    x.xx    x.xx
```

Four PNG charts are saved to the project root (see [Visualization Outputs](#visualization-outputs)).

---

## Visualization Outputs

| File | Description |
|------|-------------|
| `model_comparison.png` | Side-by-side bar charts of R², MAE, RMSE across all four models |
| `actual_vs_predicted.png` | Scatter plots of ground truth vs. predictions per model |
| `residuals_analysis.png` | Residual error patterns and distribution per model |
| `distribution_comparison.png` | Histogram overlay of actual vs. predicted soil moisture |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.0 | ML algorithms & preprocessing |
| xgboost | 2.0.3 | XGBoost regressor |
| lightgbm | 4.1.1 | LightGBM regressor |
| tensorflow | 2.14.0 | LSTM neural network |
| matplotlib | 3.8.0 | Data visualization |
| seaborn | 0.13.0 | Statistical visualization |

---

## Geographic Coverage

| Region | Climate | Role |
|--------|---------|------|
| Kitui | Semi-arid, Eastern Kenya | Training data |
| Turkana | Arid, Northern Kenya | Training data |
| Syokimau | Peri-urban, Eastern Kenya | Test + hardware validation data |

---

## Data Sources

- **TAMSAT** — satellite-derived soil moisture and rainfall estimates for Africa
- **Open Meteo** — free open-source weather API (temperature, humidity, historical data)

---

## Author

**Luke Mosoti**
- Email: lukemosoti19@gmail.com
- GitHub: [lucasmos](https://github.com/lucasmos)

---

*Last Updated: May 1, 2026*
