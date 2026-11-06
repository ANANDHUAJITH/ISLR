#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "secrets.h"

// Pin map for the common AI-Thinker ESP32-CAM module.
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WiFiServer pageServer(80);
WiFiServer streamServer(81);

bool startCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed (0x%x)\n", err);
    return false;
  }

  // Keep the factory sensor orientation. Adjust only if the module is mounted differently.
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_vflip(sensor, 0);
    sensor->set_hmirror(sensor, 0);
  }
  return true;
}

void readRequestHeaders(WiFiClient &client) {
  client.setTimeout(1500);
  client.readStringUntil('\n');  // request line
  while (client.connected()) {
    String line = client.readStringUntil('\n');
    if (line.length() <= 2) break;
  }
}

void serveStatusPage() {
  WiFiClient client = pageServer.available();
  if (!client) return;
  readRequestHeaders(client);
  const String streamUrl = String("http://") + WiFi.localIP().toString() + ":81/stream";
  String body = String("<!doctype html><html><head><meta name='viewport' content='width=device-width'>") +
    "<title>ESP32-CAM ISL Stream</title></head><body>" +
    "<h2>ESP32-CAM is online</h2><p>MJPEG stream: <a href='" + streamUrl + "'>" + streamUrl + "</a></p>" +
    "<img style='max-width:100%;height:auto' src='" + streamUrl + "'></body></html>";
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html; charset=utf-8");
  client.println("Connection: close");
  client.printf("Content-Length: %u\r\n\r\n", (unsigned)body.length());
  client.print(body);
  delay(2);
  client.stop();
}

void serveMjpegStream() {
  WiFiClient client = streamServer.available();
  if (!client) return;
  readRequestHeaders(client);
  client.setNoDelay(true);
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Cache-Control: no-cache, no-store, must-revalidate");
  client.println("Pragma: no-cache");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println();

  while (client.connected()) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      Serial.println("Camera frame capture failed");
      delay(25);
      continue;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.printf("Content-Length: %u\r\n\r\n", (unsigned)frame->len);
    const size_t sent = client.write(frame->buf, frame->len);
    client.print("\r\n");
    esp_camera_fb_return(frame);
    if (sent == 0) break;
    delay(25);  // cap stream rate and leave time for Wi-Fi/camera tasks
  }
  client.stop();
  Serial.println("Stream client disconnected");
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(500);

  if (!startCamera()) {
    Serial.println("Restarting in 5 seconds");
    delay(5000);
    ESP.restart();
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to Wi-Fi: %s", WIFI_SSID);
  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 30000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi connection timed out. Check firmware/secrets.h and reboot.");
    return;
  }

  pageServer.begin();
  streamServer.begin();
  Serial.print("Status page: http://");
  Serial.println(WiFi.localIP());
  Serial.print("MJPEG stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    return;
  }
  serveStatusPage();
  serveMjpegStream();
  delay(1);
}
