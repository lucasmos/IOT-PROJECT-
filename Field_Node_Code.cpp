/*
 * IOT PRECISION IRRIGATION: MASTER FIELD NODE
 * Integrates LoRa, SD, Sensors, I2C LCD, and Edge Impulse AI
 */

#include <SPI.h>
#include <LoRa.h>
#include <SD.h>
#include "DHT.h"
#include <LiquidCrystal_I2C.h>

// --- EDGE IMPULSE NEURAL NETWORK ---
#include <Precision-Irrigation-Control_inferencing.h>

// --- LORA SPI PINS (VSPI) ---
#define LORA_SCK 18
#define LORA_MISO 19
#define LORA_MOSI 23
#define LORA_SS 15
#define LORA_RST 32  
#define LORA_DIO0 26

// --- SD CARD SPI PINS (HSPI) ---
#define SD_SCK 14
#define SD_MISO 25  
#define SD_MOSI 13
#define SD_CS 27

// --- SENSOR & ACTUATOR PINS ---
#define DHTPIN 4
#define DHTTYPE DHT22
#define SOIL_PIN 39 
#define RELAY_PIN 5        // MOVED: Pin 21 is for LCD SDA
#define FLOW_SENSOR_PIN 33 // MOVED: Pin 22 is for LCD SCL
#define ONBOARD_LED 2  

// --- LCD DISPLAY ---
// Set the LCD address to 0x27 for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x27, 16, 2); 

// --- CALIBRATION & THRESHOLDS ---
const float AIR_VOLTAGE = 2.42;   //glued moisture sensor
const float WATER_VOLTAGE = 1.71; //glued moisture sensor
const float IRRIGATION_THRESHOLD = 35.0; // Turn on pump if moisture is below 35%
const float TARGET_LITERS = 0.15;        // Stop pump after 150ml

// --- FLOW SENSOR VARIABLES ---
volatile int pulseCount = 0;
const float CALIBRATION_FACTOR = 7.5; 

// --- DEEP SLEEP ---
const uint64_t WAKE_INTERVAL = 2ULL * 60ULL * 1000000ULL; 
const uint32_t TIME_INCREMENT_MS = 120000; 
RTC_DATA_ATTR uint32_t current_timestamp_ms = 0;

DHT dht(DHTPIN, DHTTYPE);
SPIClass spiSD(HSPI); 

// Flow sensor interrupt routine
void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

// Edge Impulse signal wrapper
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr, float *features) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void setup() {
  Serial.begin(115200);
  
  pinMode(ONBOARD_LED, OUTPUT);
  digitalWrite(ONBOARD_LED, LOW); 
  
  // Initialize Relay (Active LOW) - keep it OFF (HIGH) on boot
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);

  // Initialize Flow Sensor
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("AquaAura System");
  lcd.setCursor(0, 1);
  lcd.print("Booting up...");

  Serial.println("\n=========================================");
  Serial.printf("[SYSTEM] Waking up. Internal RTC Time: %lu ms\n", current_timestamp_ms);
  
  dht.begin();
  
  // 1. Initialize SD Card
  Serial.print("[SD CARD] Mounting... ");
  spiSD.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  pinMode(SD_CS, OUTPUT);
  bool sdOk = SD.begin(SD_CS, spiSD);
  int sdStatus = sdOk ? 1 : 0; 
  if (sdOk) Serial.println("SUCCESS!"); else Serial.println("FAILED!");

  // 2. Initialize LoRa
  Serial.print("[LORA] Initializing... ");
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (LoRa.begin(433E6)) Serial.println("SUCCESS!"); else Serial.println("FAILED!");
  LoRa.setSyncWord(0xF3);

  // 3. Read Environment 
  lcd.clear();
  lcd.print("Warming Sensors.");
  Serial.println("[SENSORS] Warming up DHT22...");
  delay(2000); 
  
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  float currentVoltage = analogReadMilliVolts(SOIL_PIN) / 1000.0;
  float relativeSat = ((AIR_VOLTAGE - currentVoltage) / (AIR_VOLTAGE - WATER_VOLTAGE)) * 100.0;
  if (relativeSat < 0) relativeSat = 0;
  if (relativeSat > 100) relativeSat = 100;

  // Display parameters on Serial Monitor and LCD
  Serial.printf("[SENSORS] Temp: %.1fC | Hum: %.1f%% | Voltage: %.2fV | Rel Sat: %.1f%%\n", t, h, currentVoltage, relativeSat);
  lcd.clear();
  lcd.setCursor(0,0); lcd.printf("T:%.1fC H:%.1f%%", t, h);
  lcd.setCursor(0,1); lcd.printf("Soil: %.1f%%", relativeSat);
  delay(3000); // Pause for presentation

  // 4. Ping Gateway for Rain Data
  bool gatewayOnline = false;
  float rain = 0.0;
  String payload = "REQ:" + String(t, 2) + "," + String(h, 2) + "," + String(relativeSat, 2) + ",0.0,0.0,0," + String(sdStatus); 

  lcd.clear(); lcd.print("Pinging Gateway");
  Serial.println("[LORA] Pinging Gateway...");
  
  for (int attempt = 1; attempt <= 3; attempt++) {
    LoRa.beginPacket(); LoRa.print(payload); LoRa.endPacket(); LoRa.receive(); 
    
    unsigned long waitStart = millis();
    while (millis() - waitStart < 10000) { 
      if (LoRa.parsePacket()) {
        String incoming = "";
        while (LoRa.available()) { incoming += (char)LoRa.read(); }
        if (incoming.startsWith("REP:")) {
          int commaIndex = incoming.indexOf(',', 4);
          rain = incoming.substring(4, commaIndex).toFloat();
          gatewayOnline = true;
          Serial.printf("  <- Gateway Ack! Rain: %.2fmm\n", rain);
          break; 
        }
      }
      delay(50); 
    }
    if (gatewayOnline) break; 
    Serial.println("  -- Timeout. Retrying...");
  }

  // 5. THE AI BRAIN: Run Edge Impulse Prediction
  lcd.clear(); lcd.print("AI Processing...");
  Serial.println("[AI] Running Edge Impulse Inference...");
  
  // Arrange features exactly as trained: [temp, hum, rain, prev_sm]
  float features[4] = { t, h, rain, relativeSat };
  
  signal_t features_signal;
  features_signal.total_length = 4;
  features_signal.get_data = [features](size_t offset, size_t length, float *out_ptr) {
      return raw_feature_get_data(offset, length, out_ptr, (float*)features);
  };

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
  
  // Extract predicted future soil moisture
  float predicted_moisture = result.classification[0].value;
  
  lcd.clear();
  lcd.setCursor(0,0); lcd.printf("Cur_SM: %.1f%%", relativeSat);
  lcd.setCursor(0,1); lcd.printf("AI_Ftr: %.1f%%", predicted_moisture);
  Serial.printf("[AI] Current SM: %.2f%% | Predicted Future SM: %.2f%%\n", relativeSat, predicted_moisture);
  delay(4000); // Pause for presentation

  // 6. IRRIGATION DECISION
  if (relativeSat < IRRIGATION_THRESHOLD && predicted_moisture < IRRIGATION_THRESHOLD) {
    Serial.println("[SYSTEM] AI Decision: IRRIGATION REQUIRED.");
    lcd.clear(); lcd.print("Watering Crops!");
    
    // Attach interrupt and turn pump ON
    pulseCount = 0;
    attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), pulseCounter, FALLING);
    digitalWrite(RELAY_PIN, LOW); // Active Low = Pump ON
    
    float totalLiters = 0.0;
    while (totalLiters < TARGET_LITERS) {
      noInterrupts();
      totalLiters = pulseCount / (CALIBRATION_FACTOR * 60.0); // Convert pulses to absolute Liters
      interrupts();
      
      lcd.setCursor(0,1);
      lcd.printf("Volume: %.2f L", totalLiters);
      delay(200);
    }
    
    // Stop Pump
    digitalWrite(RELAY_PIN, HIGH);
    detachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN));
    
    lcd.clear();
    lcd.print("Watering Done.");
    Serial.println("[SYSTEM] Target volume reached. Pump OFF.");
    delay(3000);
  } else {
    Serial.println("[SYSTEM] AI Decision: No irrigation needed.");
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("No Water Needed");
    lcd.setCursor(0,1); lcd.print("Safe by AI.");
    delay(3000);
  }

  // 7. DATA LOGGING & STORE-AND-FORWARD SYNC
  String csvData = String(current_timestamp_ms) + "," + String(t, 2) + "," + String(h, 2) + "," + String(rain, 2) + "," + String(relativeSat, 2) + ",0,0,0";
  
  if (sdOk) {
    // 7A. Always write to Master CSV
    Serial.print("[SD CARD] Writing to Master CSV... ");
    File dataFile = SD.open("/syokimau_data.csv", FILE_APPEND);
    if (dataFile) { 
      dataFile.println(csvData); 
      dataFile.close(); 
      Serial.println("Done.");
    } else { 
      Serial.println("Failed!"); 
    }

    // 7B. Handle Offline / Online Logic
    if (!gatewayOnline) {
      Serial.println("[LORA] Gateway offline. Appending to Backlog CSV...");
      File backlogFile = SD.open("/backlog.csv", FILE_APPEND);
      if (backlogFile) {
        backlogFile.println(csvData);
        backlogFile.close();
        Serial.println("[SD CARD] Data safely saved to backlog.");
        
        // VISUAL DIAGNOSTIC: 10 Rapid Blue Flashes for Offline Saving
        lcd.clear();
        lcd.print("Gateway Offline!");
        lcd.setCursor(0,1);
        lcd.print("Saved to Backlog");
        
        for (int i = 0; i < 10; i++) {
          digitalWrite(ONBOARD_LED, HIGH);
          delay(100);
          digitalWrite(ONBOARD_LED, LOW);
          delay(100);
        }
      }
    } else {
      // Gateway is online! Check if we have historical backlog data to push
      if (SD.exists("/backlog.csv")) {
        Serial.println("[SYNC] Gateway found! Pushing historical backlog data...");
        lcd.clear();
        lcd.print("Syncing Backlog");
        
        File bklg = SD.open("/backlog.csv", FILE_READ);
        if (bklg) {
          int linesSynced = 0;
          while (bklg.available()) {
            String line = bklg.readStringUntil('\n');
            line.trim();
            if (line.length() > 0) {
              String hisPayload = "HIS:" + line; 
              LoRa.beginPacket();
              LoRa.print(hisPayload);
              LoRa.endPacket();
              linesSynced++;
              Serial.printf("  -> Uploading row %d...\n", linesSynced);
              
              lcd.setCursor(0,1);
              lcd.printf("Row: %d sent", linesSynced);
              delay(350); // Pause so Gateway has time to process and push to Blynk
            }
          }
          bklg.close();
          SD.remove("/backlog.csv"); // Clear the backlog once completely sent
          Serial.printf("[SYNC] Complete. %d historical records pushed.\n", linesSynced);
          
          // VISUAL DIAGNOSTIC: 1 Solid Flash for Successful Sync
          digitalWrite(ONBOARD_LED, HIGH);
          delay(1500);
          digitalWrite(ONBOARD_LED, LOW);
        }
      }
    }
  }

  // 8. Sleep
  lcd.clear();
  lcd.print("Sleeping...");
  current_timestamp_ms += TIME_INCREMENT_MS; 
  Serial.println("[SYSTEM] Going to Deep Sleep.");
  Serial.println("=========================================\n");
  
  delay(2000); // Let the audience read the final LCD message
  lcd.noBacklight(); // Turn off LCD backlight to save power
  
  LoRa.sleep(); 
  esp_sleep_enable_timer_wakeup(WAKE_INTERVAL);
  Serial.flush(); 
  esp_deep_sleep_start();
}

void loop() {}