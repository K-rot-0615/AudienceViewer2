#include <ESP8266WiFi.h>
#include <WiFiUDP.h>

const char* ssid = "koblab-1";
const char* password = "koblabfms1116";

WiFiUDP Udp;
unsigned int localUdpPort = 8888;
char incomingPacket[255];
char* low = "LOW";
char* high = "HIGH";
char* average_high = "average_HIGH";
char replyPacket[] = "Hi there! Got the message : -)";


void setup(){

  pinMode(4, OUTPUT);

  Serial.begin(115200);
  Serial.println();

  Serial.printf("connecting to %s ", ssid);
  WiFi.begin(ssid, password);

  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  //Serial.println(" connected");

  Udp.begin(localUdpPort);
  //Serial.printf("Now listening at IP %s, UDP port %d\n", WiFi.localIP().toString().c_str(), localUdpPort);
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());
}

void loop(){
 int packetSize = Udp.parsePacket();
 if(packetSize){
  //Serial.printf("Received %d bytes from %s, port %d\n", packetSize, Udp.remoteIP().toString().c_str(), Udp.remotePort());
  Udp.read(incomingPacket, 255);

  Serial.println(incomingPacket);

  if(strcmp(incomingPacket, low) == 0){
    Serial.println("LOW!!!!!!!");
    digitalWrite(4, LOW);
  }
  
  if(strcmp(incomingPacket, high) == 0){
    Serial.println("HIGH!!!!!!");
    digitalWrite(4, HIGH);
  }

  if(strcmp(incomingPacket, average_high) == 0){
    Serial.println("averageishigh");
    for(int i=0; i<3; i++){
      digitalWrite(4, HIGH);
      delay(100);
      digitalWrite(4, LOW);
      delay(100);
    }
  }

  Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
  Udp.write(replyPacket);
  Udp.endPacket();
 }
}
