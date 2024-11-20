#define TRIG_PIN 9   // Pino de disparo (Trig)
#define ECHO_PIN 10  // Pino de recepção (Echo)

void setup() {
  Serial.begin(9600);  // Inicializa a comunicação serial
  pinMode(TRIG_PIN, OUTPUT);  // Configura o pino Trig como saída
  pinMode(ECHO_PIN, INPUT);   // Configura o pino Echo como entrada
}

void loop() {
  // Envia um pulso de 10 microssegundos para o pino TRIG
  digitalWrite(TRIG_PIN, LOW);  
  delayMicroseconds(2);  // Garante que o pino TRIG fique LOW por 2 microssegundos
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10); // O pulso dura 10 microssegundos
  digitalWrite(TRIG_PIN, LOW);

  // Agora mede o tempo que o sinal levou para retornar
  long duration = pulseIn(ECHO_PIN, HIGH);  // Recebe o tempo de resposta

  // Calcula a distância em centímetros
  long distance = (duration / 2) / 29.1;  // Fórmula para calcular a distância

  // Exibe a distância no Serial Monitor
  Serial.print("Distância: ");
  Serial.print(distance);
  Serial.println(" cm");

  delay(500);  // Espera meio segundo antes de medir novamente
}
