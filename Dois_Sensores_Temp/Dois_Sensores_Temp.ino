#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_BUS 10  // Define o pino onde os sensores estão conectados (exemplo: pino digital 10)

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(9600);        // Inicializa a comunicação serial
  sensors.begin();           // Inicializa o barramento OneWire e os sensores de temperatura

  Serial.println("Procurando por sensores...");

  // Verifica quantos sensores estão conectados
  int deviceCount = sensors.getDeviceCount();
  Serial.print("Sensores encontrados: ");
  Serial.println(deviceCount);

  if (deviceCount < 2) {
    Serial.println("Menos de 2 sensores encontrados!");
  } else {
    Serial.println("2 sensores encontrados!");
  }
}

void loop() {
  sensors.requestTemperatures();  // Solicita a leitura de temperatura de todos os sensores

  // Lê e exibe a temperatura dos dois primeiros sensores
  for (int i = 0; i < 2; i++) {
    Serial.print("Sensor ");
    Serial.print(i + 1);              // Exibe o número do sensor
    Serial.print(": ");
    Serial.print(sensors.getTempCByIndex(i));  // Lê a temperatura em Celsius do sensor
    Serial.println(" °C");
  }

  delay(1000);  // Espera 1 segundo antes de fazer uma nova leitura
}
