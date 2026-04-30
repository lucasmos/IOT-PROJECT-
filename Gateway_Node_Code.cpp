#define BLYNK_TEMPLATE_ID "TMPL2aBRQHFV0"
#define BLYNK_TEMPLATE_NAME "Irrigation Node"
#define BLYNK_AUTH_TOKEN "4KHJciklWut8LKN8Mx3C3_4W3JRrC50N"

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BlynkSimpleEsp32.h>
#include <SPI.h>
#include <LoRa.h>
#include "time.h"
#include <esp_task_wdt.h>

// --- CREDENTIALS & API ---
const char* ssid = "NAMALUKI";
const char* pass = "suchcharacter10";
const char* api_url = "http://api.open-meteo.com/v1/forecast?latitude=-1.37&longitude=36.93&current=rain&models=ecmwf_ifs&timezone=auto";

// --- LORA & LED PINS ---
#define SCK 18
#define MISO 19
#define MOSI 23
#define SS 15
#define RST 32
#define DIO0 26
#define ONBOARD_LED 2

// --- INTERRUPT VARIABLES ---
volatile bool loraPacketReceived = false;
volatile int currentRssi = 0;
String incomingData = "";

// --- THE INTERRUPT SERVICE ROUTINE (ISR) ---
void IRAM_ATTR onReceive(int packetSize) {
  if (packetSize == 0) return;

  incomingData = "";
  while (LoRa.available()) {
    incomingData += (char)LoRa.read();
  }
  currentRssi = LoRa.packetRssi();
  loraPacketReceived = true;
}

void setup() {
  Serial.begin(115200);

  pinMode(ONBOARD_LED, OUTPUT);
  digitalWrite(ONBOARD_LED, LOW);

  Serial.println("\n\n=========================================");
  Serial.println("[SYSTEM] Gateway Node Booting up...");

  Serial.print("[WIFI] Connecting to SSID: ");
  Serial.print(ssid);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[WIFI] Connected Successfully!");

  Serial.println("[API] Syncing internal clock with pool.ntp.org...");
  configTime(10800, 0, "pool.ntp.org");

  Serial.println("[BLYNK] Attempting connection to cloud servers...");
  Blynk.config(BLYNK_AUTH_TOKEN);
  if (Blynk.connect()) {
    Serial.println("[BLYNK] Connected to Blynk Cloud successfully!");
  } else {
    Serial.println("[BLYNK] WARNING: Timeout connecting to Blynk Cloud.");
  }

  Serial.print("[LORA] Initializing Radio module... ");
  SPI.begin(SCK, MISO, MOSI, SS);
  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(433E6)) {
    Serial.println("FAILED! System halted. Triggering 10-flash error sequence.");
    for (int i = 0; i < 10; i++) {
      digitalWrite(ONBOARD_LED, HIGH);
      delay(100);
      digitalWrite(ONBOARD_LED, LOW);
      delay(100);
    }
    while (1)
      ;
  }
  Serial.println("SUCCESS!");
  LoRa.setSyncWord(0xF3);

  LoRa.onReceive(onReceive);
  LoRa.receive();

  Serial.println("[SYSTEM] Gateway fully online. Listening via Hardware Interrupts...");

  // Initialize the Watchdog Timer
  // We use this simpler format for the Arduino IDE's ESP32 Core
  esp_task_wdt_config_t wdt_config = {
    .timeout_ms = 15000,
    .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,
    .trigger_panic = true
  };
  // We add NULL to simply subscribe the main loop to the existing background WDT
  esp_task_wdt_add(NULL);

  Serial.println("=========================================\n");
}

void loop() {
  esp_task_wdt_reset();

  // --- Wi-Fi Keep-Alive ---
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WIFI] Connection lost! Reconnecting...");
    WiFi.disconnect();
    WiFi.begin(ssid, pass);
    unsigned long startAttemptTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) {
      delay(500);
      esp_task_wdt_reset();
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("[WIFI] Connection restored!");
      Blynk.connect();
    }
  }

  if (WiFi.status() == WL_CONNECTED) { Blynk.run(); }

  // --- LoRa Interrupt Handling ---
  if (loraPacketReceived) {
    Serial.println("\n-----------------------------------------");
    Serial.printf("[LORA RX] Packet Received (Signal: %d dBm): %s\n", currentRssi, incomingData.c_str());

    // CASE A: Live Data Request
    if (incomingData.startsWith("REQ:")) {
      Serial.println("[PARSER] Detected Live Sensor Data Request.");
      String dataStr = incomingData.substring(4);
      float t, h, m, battV, sysV;
      int isCharging, sdStatus;

      if (sscanf(dataStr.c_str(), "%f,%f,%f,%f,%f,%d,%d", &t, &h, &m, &battV, &sysV, &isCharging, &sdStatus) == 7) {

        Serial.println("[API] Fetching real-time rainfall data...");
        unsigned long apiTimerStart = millis();
        float rain = fetchRainfall();
        unsigned long apiDuration = millis() - apiTimerStart;
        Serial.printf("[API] Weather fetched in %lu ms. Rain: %.2f mm\n", apiDuration, rain);

        String timeStr = getLocalTimeStr();

        if (WiFi.status() == WL_CONNECTED) {
          Serial.print("[BLYNK] Pushing live data to dashboard... ");
          Blynk.virtualWrite(V0, t);
          Blynk.virtualWrite(V1, h);
          Blynk.virtualWrite(V2, rain);
          Blynk.virtualWrite(V3, m);
          Blynk.virtualWrite(V4, battV);
          Blynk.virtualWrite(V5, sysV);
          Blynk.virtualWrite(V6, isCharging);
          Blynk.virtualWrite(V7, sdStatus);
          Serial.println("Done.");

          digitalWrite(ONBOARD_LED, HIGH);
          delay(1000);
          digitalWrite(ONBOARD_LED, LOW);

        } else {
          Serial.println("[BLYNK] Skipped push (Wi-Fi disconnected).");
        }

        String replyMsg = "REP:" + String(rain, 2) + "," + timeStr;
        delay(50);
        LoRa.beginPacket();
        LoRa.print(replyMsg);
        LoRa.endPacket();
        Serial.printf("[LORA TX] Sent Acknowledgement: %s\n", replyMsg.c_str());

        LoRa.receive();
      } else {
        Serial.println("[PARSER] ERROR: Invalid data format.");
        LoRa.receive();
      }
    }

    // CASE B: Historical Backlog Sync
    else if (incomingData.startsWith("HIS:")) {
      Serial.println("[PARSER] Detected Historical Sync Payload.");
      String dataStr = incomingData.substring(4);
      uint32_t ts;
      float t, h, r, m, battV, sysV;
      int isCharging;

      if (sscanf(dataStr.c_str(), "%lu,%f,%f,%f,%f,%f,%f,%d", &ts, &t, &h, &r, &m, &battV, &sysV, &isCharging) == 8) {
        if (WiFi.status() == WL_CONNECTED) {
          Blynk.virtualWrite(V0, t);
          Blynk.virtualWrite(V1, h);
          Blynk.virtualWrite(V2, r);
          Blynk.virtualWrite(V3, m);
          Blynk.virtualWrite(V4, battV);
          Blynk.virtualWrite(V5, sysV);
          Blynk.virtualWrite(V6, isCharging);
          Serial.printf("[BLYNK] Timestamp %lu pushed to cloud.\n", ts);

          digitalWrite(ONBOARD_LED, HIGH);
          delay(100);
          digitalWrite(ONBOARD_LED, LOW);
        }
      }
      LoRa.receive();
    }
    Serial.println("-----------------------------------------");

    loraPacketReceived = false;
  }
}

// --- HELPER FUNCTIONS ---
float fetchRainfall() {
  float rainValue = 0.0;
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(api_url);

    // CRITICAL FIX: The 5-second timeout prevents the ESP32 from freezing and rebooting!
    http.setTimeout(5000);

    if (http.GET() > 0) {
      String payload = http.getString();

      // Reverted back to v6 syntax for the Arduino IDE
      StaticJsonDocument<128> filter;
      filter["current"]["rain"] = true;
      StaticJsonDocument<512> doc;
      if (!deserializeJson(doc, payload, DeserializationOption::Filter(filter))) {
        rainValue = doc["current"]["rain"];
      }
    }
    http.end();
  }
  return rainValue;
}

String getLocalTimeStr() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) { return "Time_Error"; }
  char timeStringBuff[10];
  strftime(timeStringBuff, sizeof(timeStringBuff), "%H:%M:%S", &timeinfo);
  return String(timeStringBuff);
}