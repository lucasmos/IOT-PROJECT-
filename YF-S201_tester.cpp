// --- PIN DEFINITION ---
#define FLOW_SENSOR_PIN 22

// --- VARIABLES ---
// 'volatile' tells the ESP32 this variable will change instantly via an interrupt
volatile int pulseCount = 0;
float flowRate = 0.0;
unsigned long previousMillis = 0;

// The standard calibration factor for the YF-S201 is 7.5.
// This means 1 Liter per minute generates 7.5 pulses per second.
const float CALIBRATION_FACTOR = 7.5; 

// --- THE INTERRUPT FUNCTION ---
// This fires instantly every time the sensor's internal magnet spins
void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

void setup() {
  Serial.begin(115200);
  
  // Set the pin as an input. INPUT_PULLUP helps stabilize the 3.3V logic signal.
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  
  // Attach the hardware interrupt to the pin
  // "FALLING" means it triggers every time the pulse drops from 3.3V to 0V
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), pulseCounter, FALLING);
  
  Serial.println("\n--- Water Flow Sensor 3.3V Test ---");
  Serial.println("Blow into the sensor to test!");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Process the data exactly once every second (1000 ms)
  if (currentMillis - previousMillis >= 1000) {
    
    // 1. Temporarily stop interrupts so we can safely read the pulse count
    noInterrupts();
    int currentPulses = pulseCount;
    pulseCount = 0; // Reset for the next second
    interrupts();   // Turn interrupts back on immediately
    
    // 2. Calculate the Flow Rate in Liters per Minute
    // Math: (Pulses in this second / 7.5) = Liters/Minute
    flowRate = ((1000.0 / (currentMillis - previousMillis)) * currentPulses) / CALIBRATION_FACTOR;
    
    // Update the timer
    previousMillis = currentMillis;

    // 3. Print to the Serial Monitor
    if (currentPulses > 0) {
      Serial.printf("Pulses: %d  |  Flow Rate: %.2f L/min\n", currentPulses, flowRate);
    } else {
      Serial.println("Waiting for water flow...");
    }
  }
}