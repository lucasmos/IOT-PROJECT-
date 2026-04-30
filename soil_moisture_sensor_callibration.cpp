/**
 * Capacitive Soil Moisture Sensor Calibration
 * ESP32-DEVKIT
 *
 * How to use:
 *  1. Upload this sketch to your ESP32.
 *  2. Open Serial Monitor at 115200 baud.
 *  3. Follow the prompts to record DRY and WET voltage readings.
 *  4. Copy the printed calibration constants into your main project.
 *
 * Wiring:
 *  Sensor AOUT  →  GPIO 34 (or any ADC1 pin: 32–39)
 *  Sensor VCC   →  3.3 V
 *  Sensor GND   →  GND
 *
 * Notes:
 *  - Use ADC1 pins (32–39) only. ADC2 is disabled when Wi-Fi is active.
 *  - ESP32 ADC has known non-linearity; this sketch applies the standard
 *    two-point linear correction automatically.
 *  - Leave the sensor in each medium for at least 30 seconds before
 *    pressing a key so the reading can stabilise.
 */

// ─── Pin & ADC settings ──────────────────────────────────────────────────────
#define SENSOR_PIN      39        // Analog input pin (ADC1)
#define ADC_RESOLUTION  12        // 12-bit → 0–4095
#define ADC_VREF        3.3f      // Reference voltage (V)
#define NUM_SAMPLES     64        // Samples to average per reading
#define SAMPLE_DELAY_MS 5         // Delay between each sample (ms)

// ─── Calibration storage (updated at runtime) ────────────────────────────────
float voltageAir  = 0.0f;   // Raw voltage with sensor in open air (dry reference)

float voltageWater = 0.0f;  // Raw voltage with sensor fully submerged (wet reference)

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Read ADC NUM_SAMPLES times and return the averaged raw count. */
uint32_t readAveragedRaw() {
  uint64_t sum = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum += analogRead(SENSOR_PIN);
    delay(SAMPLE_DELAY_MS);
  }
  return (uint32_t)(sum / NUM_SAMPLES);
}

/** Convert a 12-bit ADC count to voltage (V). */
float rawToVoltage(uint32_t raw) {
  return (raw / 4095.0f) * ADC_VREF;
}

/**
 * Map voltage to moisture percentage using the two calibration points.
 * Capacitive sensors give HIGHER voltage when DRY and LOWER when WET.
 */
float voltageToMoisture(float voltage) {
  if (voltageAir == voltageWater) return -1.0f;   // Division by zero guard
  float pct = (voltageAir - voltage) / (voltageAir - voltageWater) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}

/** Block until a character is received on Serial. */
void waitForKey() {
  while (Serial.available()) Serial.read();  // flush
  while (!Serial.available()) delay(100);
  while (Serial.available()) Serial.read();  // flush again
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(ADC_RESOLUTION);
  analogSetAttenuation(ADC_11db);   // Full-scale ≈ 3.3 V with 11 dB attenuation

  Serial.println();
  Serial.println("╔══════════════════════════════════════════════╗");
  Serial.println("║  Capacitive Soil Moisture Sensor Calibrator  ║");
  Serial.println("║  ESP32-DEVKIT  •  115200 baud                ║");
  Serial.println("╚══════════════════════════════════════════════╝");
  Serial.println();

  // ── Step 1: DRY calibration ──────────────────────────────────────────────
  Serial.println("STEP 1 — DRY calibration");
  Serial.println("  Hold the sensor in open air (completely dry).");
  Serial.println("  Press ENTER when ready…");
  waitForKey();

  uint32_t dryRaw = readAveragedRaw();
  voltageAir = rawToVoltage(dryRaw);

  Serial.printf("  ✔  DRY  raw = %4u  |  voltage = %.4f V\n\n", dryRaw, voltageAir);

  // ── Step 2: WET calibration ──────────────────────────────────────────────
  Serial.println("STEP 2 — WET calibration");
  Serial.println("  Submerge the sensor in water up to the max-fill line.");
  Serial.println("  Wait ~30 s for the reading to settle, then press ENTER…");
  waitForKey();

  uint32_t wetRaw = readAveragedRaw();
  voltageWater = rawToVoltage(wetRaw);

  Serial.printf("  ✔  WET  raw = %4u  |  voltage = %.4f V\n\n", wetRaw, voltageWater);

  // ── Print calibration summary ────────────────────────────────────────────
  Serial.println("══════════════════════════════════════════════════");
  Serial.println("  CALIBRATION COMPLETE — copy these constants");
  Serial.println("  into your main sketch:");
  Serial.println();
  Serial.printf("  #define MOISTURE_VOLTAGE_DRY   %.4ff\n", voltageAir);
  Serial.printf("  #define MOISTURE_VOLTAGE_WET   %.4ff\n", voltageWater);
  Serial.println();

  if (voltageAir < voltageWater) {
    Serial.println("  ⚠  WARNING: DRY voltage is LOWER than WET voltage.");
    Serial.println("     This is unusual for a capacitive sensor. Check:");
    Serial.println("     • Wiring (VCC/GND swapped?)");
    Serial.println("     • Sensor fully dry during step 1?");
    Serial.println("     • Correct ADC pin (use GPIO 32–39 only)?");
  } else {
    float span = voltageAir - voltageWater;
    Serial.printf("  Voltage span: %.4f V", span);
    if (span < 0.3f)
      Serial.print("  ← Low span; re-calibrate for better accuracy");
    Serial.println();
  }

  Serial.println("══════════════════════════════════════════════════");
  Serial.println();
  Serial.println("Now reading moisture continuously (press RST to re-calibrate):");
  Serial.println();
}

// ─── Loop ────────────────────────────────────────────────────────────────────
void loop() {
  uint32_t raw      = readAveragedRaw();
  float    voltage  = rawToVoltage(raw);
  float    moisture = voltageToMoisture(voltage);

  // Simple ASCII bar graph (50 chars wide)
  int bars = (int)(moisture / 2.0f);   // 0–50
  char bar[52] = {0};
  for (int i = 0; i < 50; i++) bar[i] = (i < bars) ? '#' : '-';

  Serial.printf("Raw: %4u  |  %.4f V  |  Moisture: %5.1f %%  [%s]\n",
                raw, voltage, moisture, bar);

  delay(1000);
}
