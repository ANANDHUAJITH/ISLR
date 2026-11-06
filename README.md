# ISL ESP32-CAM Letter-to-Speech

A small prototype that streams video from an AI-Thinker ESP32-CAM over Wi-Fi, extracts hand landmarks on a nearby computer with MediaPipe, recognizes fingerspelled ISL alphabet letters, joins them into words, and speaks each word locally.

```text
AI-Thinker ESP32-CAM --HTTP/MJPEG over Wi-Fi--> Python host
                                                MediaPipe Hands landmarks
                                                trained ISL letter SVM
                                                temporal smoothing + word pause
                                                local text-to-speech
```

This is **fingerspelling**, not a full sign-language translator. It recognizes still hand shapes from the A–Z/`none` training set; it does not infer ISL grammar or translate whole-body/moving signs. Letters are concatenated in order. A quiet pause marks the end of a word, and the computer speaks that word. Repeating the same letter (for example the two Ls in “HELLO”) requires lowering the hand briefly between them.

## What is in this repository

- `firmware/esp32_cam_stream/esp32_cam_stream.ino` — AI-Thinker ESP32-CAM Wi-Fi setup and MJPEG server.
- `host/live_letter_speaker.py` — connects to the stream, runs MediaPipe HandLandmarker and the trained classifier, smooths predictions, buffers letters, and speaks completed words.
- `host/train_classifier.py` — extracts MediaPipe landmarks from class folders and trains/evaluates a replacement classifier.
- `models/hand_landmarker.task` — MediaPipe hand-landmarker asset used at runtime.
- `models/islr_mediapipe_landmarks_svm.joblib` — trained 27-label classifier (`A`–`Z`, `none`).
- `reports/` — held-out evaluation files from the included model.

## Hardware and network

This firmware targets the common **AI-Thinker ESP32-CAM** pin layout. Use a 2.4 GHz Wi-Fi network; the ESP32-CAM and the computer running Python must be on the same LAN. The camera sends frames to the computer; inference and speech happen on the computer, not on the ESP32.

## Flash the ESP32-CAM

1. Install Arduino IDE and the Espressif ESP32 Arduino board package (ESP32 Arduino core 3.x recommended).
2. Select **AI Thinker ESP32-CAM** as the board. Connect a USB-to-serial adapter or ESP32-CAM programmer; many boards require GPIO0 connected to GND during flashing. Use a stable 5 V supply.
3. In `firmware/esp32_cam_stream/`, copy `secrets.example.h` to `secrets.h`, then enter the 2.4 GHz Wi-Fi name and password in `secrets.h`. That file is ignored by Git.
4. Open `esp32_cam_stream.ino`, select the correct serial port, and upload. If required, hold BOOT while upload starts. After flashing, remove the GPIO0-to-GND jumper and reset the board.
5. Open Serial Monitor at **115200 baud**. Copy the printed stream URL, which looks like `http://192.168.1.42:81/stream`. The status page is at `http://192.168.1.42/`.

The firmware uses VGA on boards with PSRAM and QVGA otherwise. If your camera is a different ESP32-CAM variant, update the pin map at the top of the sketch before flashing.

## Run the Python recognizer

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python host/live_letter_speaker.py --url http://192.168.1.42:81/stream
```

Replace the example address with the URL printed by your ESP32-CAM. Keep the model files in `models/` (the script uses those paths by default).

Text-to-speech runs on the computer using `pyttsx3` and the operating system's local speech engine. On Ubuntu/Debian, install an eSpeak voice engine if needed:

```bash
sudo apt-get install espeak-ng
```

If speech setup is unavailable, the recognizer still prints words; run with `--no-tts` to disable speech intentionally. No cloud speech service is used.

### Word-building controls

The host requires a stable letter prediction across several frames before appending it. A sustained pose is appended once, not repeatedly. Lower the hand long enough to re-arm the same letter. A continuous no-letter pause ends the buffered word and sends it to local TTS.

Useful options:

```bash
python host/live_letter_speaker.py --url http://192.168.1.42:81/stream \
  --word-pause 1.7 --release-pause 0.35 --fps 12
```

- `--word-pause` — seconds without a recognized letter before the current word is printed and spoken.
- `--release-pause` — no-letter time required before the same letter can be appended again.
- `--window` / `--stable-frames` — temporal smoothing and stability threshold.
- `--fps` — maximum inference rate.
- `--no-tts` — print completed words without speaking them.

## Retrain on another dataset

Expected layout is one directory per label, for example `dataset/A/*.jpg` through `dataset/Z/*.jpg` and `dataset/none/*.jpg`. The same MediaPipe hand model and landmark normalization are used during training and inference.

```bash
python host/train_classifier.py \
  --data /path/to/islr_data_sample \
  --hand-model models/hand_landmarker.task \
  --output-dir models/new-training
```

Use the classifier path option at runtime to select that run's model:

```bash
python host/live_letter_speaker.py --url http://192.168.1.42:81/stream \
  --classifier models/new-training/islr_mediapipe_landmarks_svm.joblib
```

## Current model evaluation

The bundled classifier was trained from the public [`ANANDHUAJITH/ISLR` sample](https://github.com/ANANDHUAJITH/ISLR/tree/main/islr_data_sample): 593 images across 27 labels. Its stratified 25% holdout contains 149 images. It scored **96.64% accuracy** and **0.9676 macro-F1** on that split; MediaPipe detected a hand in 98.3% of the full sample. Full per-class metrics and the confusion matrix are in `reports/`.

The holdout is small and likely shares acquisition conditions with training data; real-world accuracy can be lower. For a reliable deployment, collect more samples across people, lighting, distances, camera rotations, and backgrounds, and test on people not seen during training. The static-image data cannot teach temporal gestures (notably motion-based J/Z); this repository treats each as a static hand pose.

## Troubleshooting

- **Stream URL not reachable:** verify both devices are on the same LAN, check the IP in Serial Monitor, and confirm the URL ends in `:81/stream`. Guest Wi-Fi or router client isolation may block traffic between devices.
- **Camera init failure or brownout:** use a stable 5 V supply and select the correct AI-Thinker board/pin map.
- **No letters appear:** position one hand clearly in frame, improve lighting, and test with hand signs similar to the training sample. The trained classifier is not guaranteed to generalize to a different signing style/background.
- **Speech is silent:** install an OS speech engine (e.g. `espeak-ng` on Linux) and verify `pyttsx3` installed successfully.
