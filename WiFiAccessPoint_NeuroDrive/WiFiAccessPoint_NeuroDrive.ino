#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

/* Defina as credenciais para o ponto de acesso Wi-Fi */
const char *ssid = "NeuroDrive";
const char *password = "Raptor2025";

ESP8266WebServer server(80);
unsigned long lastBlink = 0;
bool ledState = LOW;
const int ledPin = LED_BUILTIN; // LED embutido no ESP8266, geralmente GPIO2

void setup() {
  delay(1000);
  Serial.begin(115200);
  Serial.println();
  Serial.print("Configuring access point...");

  // Configura o pino LED_BUILTIN como saída e começa apagado (ativo baixo)
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH); // Desliga o LED inicialmente (HIGH = desligado no ESP8266)

  // Configura o ponto de acesso Wi-Fi
  WiFi.softAP(ssid, password);

  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(myIP);

  // Configura o servidor web para responder na rota raiz
  server.on("/", []() {
    server.send(200, "text/html", "<h1>You are connected</h1>");
  });
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();

  // Obtém o número de dispositivos conectados ao ponto de acesso
  int numClients = WiFi.softAPgetStationNum();

  if (numClients > 0) {
    // Se há pelo menos um cliente conectado, mantém o LED aceso
    digitalWrite(ledPin, LOW); // Liga o LED (ativo baixo)
  } else {
    // Se não há clientes conectados, faz o LED piscar
    unsigned long currentMillis = millis();
    if (currentMillis - lastBlink >= 500) { // Pisca a cada 500 ms
      lastBlink = currentMillis;
      ledState = !ledState; // Alterna o estado do LED
      digitalWrite(ledPin, ledState ? LOW : HIGH); // LOW para acender, HIGH para apagar
    }
  }
}
