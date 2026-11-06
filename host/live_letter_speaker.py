#!/usr/bin/env python3
"""Read ESP32-CAM MJPEG, classify static ISL letters, build words, and speak them."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from io import BytesIO
from pathlib import Path
import queue
import threading
import time
import urllib.request

import joblib
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def landmark_feature(result) -> np.ndarray:
    """Must match host/train_classifier.py feature normalization exactly."""
    if not result.hand_landmarks:
        return np.zeros(64, dtype=np.float32)
    index = 0
    if result.handedness:
        index = max(range(len(result.handedness)), key=lambda i: result.handedness[i][0].score)
    points = np.asarray([[p.x, p.y, p.z] for p in result.hand_landmarks[index]], dtype=np.float32)
    points -= points[0].copy()  # wrist anchor
    if result.handedness and len(result.handedness) > index:
        if result.handedness[index][0].category_name.lower() == "left":
            points[:, 0] *= -1.0
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale > 1e-6:
        points /= scale
    return np.concatenate([points.reshape(-1), np.asarray([1.0], dtype=np.float32)])


def mjpeg_frames(url: str):
    """Yield RGB numpy frames from an HTTP multipart MJPEG endpoint."""
    request = urllib.request.Request(url, headers={"User-Agent": "ISL-ESP32-CAM-Speech/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        buffer = bytearray()
        while True:
            chunk = response.read(8192)
            if not chunk:
                return
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 1:
                        del buffer[:-1]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start > 0:
                        del buffer[:start]
                    break
                jpeg = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                try:
                    image = Image.open(BytesIO(jpeg)).convert("RGB")
                    # Resize large frames for smoother on-device hand-landmark inference.
                    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    yield np.asarray(image)
                except Exception as exc:
                    print(f"Skipping invalid JPEG frame: {exc}")


class SpeechWorker(threading.Thread):
    """Keep local speech synthesis off the camera-processing loop."""
    def __init__(self, enabled: bool):
        super().__init__(daemon=True)
        self.enabled = enabled and pyttsx3 is not None
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.engine = None

    def run(self):
        if not self.enabled:
            return
        try:
            self.engine = pyttsx3.init()
        except Exception as exc:
            print(f"Text-to-speech unavailable ({exc}); words will still print in the terminal.")
            self.enabled = False
            return
        while True:
            text = self.messages.get()
            try:
                if text is None:
                    return
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as exc:
                print(f"Speech error: {exc}")
            finally:
                self.messages.task_done()

    def speak(self, text: str):
        if self.enabled:
            self.messages.put(text)

    def close(self):
        if self.enabled and self.is_alive():
            self.messages.put(None)
            self.messages.join()
            self.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="ESP32-CAM MJPEG URL, e.g. http://192.168.1.42:81/stream")
    parser.add_argument("--classifier", type=Path, default=Path(__file__).resolve().parents[1] / "models/islr_mediapipe_landmarks_svm.joblib")
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).resolve().parents[1] / "models/hand_landmarker.task")
    parser.add_argument("--word-pause", type=float, default=1.7, help="Seconds without a recognized letter before speaking the buffered word")
    parser.add_argument("--release-pause", type=float, default=0.35, help="No-letter time needed to re-arm the same letter")
    parser.add_argument("--window", type=int, default=5, help="Frame smoothing window")
    parser.add_argument("--stable-frames", type=int, default=4, help="Matching labels needed in the smoothing window")
    parser.add_argument("--fps", type=float, default=12.0, help="Maximum inference rate")
    parser.add_argument("--no-tts", action="store_true", help="Print completed words without speaking them")
    args = parser.parse_args()

    if args.window < 1 or not 1 <= args.stable_frames <= args.window:
        parser.error("Require 1 <= --stable-frames <= --window")
    if not args.classifier.is_file() or not args.hand_model.is_file():
        parser.error("Classifier or hand-landmarker model file is missing")

    bundle = joblib.load(args.classifier)
    model = bundle["model"]
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
    speaker = SpeechWorker(enabled=not args.no_tts)
    speaker.start()
    if not args.no_tts and pyttsx3 is None:
        print("pyttsx3 is not installed; speech disabled. Install requirements.txt or use --no-tts.")

    history = deque(maxlen=args.window)
    word: list[str] = []
    last_emitted = None
    no_letter_since = None
    frame_period = 1.0 / max(args.fps, 0.1)
    print(f"Connecting to {args.url}")
    print("Show one static letter at a time. Briefly lower the hand between repeated letters.")
    print(f"A pause of {args.word_pause:.1f}s ends a word; press Ctrl+C to stop.")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            for rgb in mjpeg_frames(args.url):
                start = time.monotonic()
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                feature = landmark_feature(result).reshape(1, -1)
                history.append(str(model.predict(feature)[0]))
                counts = Counter(history)
                candidate, occurrences = counts.most_common(1)[0]
                stable = candidate if len(history) == args.window and occurrences >= args.stable_frames else None
                now = time.monotonic()

                if stable is not None and stable != "none":
                    no_letter_since = None
                    if stable != last_emitted:
                        word.append(stable)
                        last_emitted = stable
                        print(f"\nLetter: {stable}    Buffer: {''.join(word)}", flush=True)
                else:
                    if no_letter_since is None:
                        no_letter_since = now
                    if now - no_letter_since >= args.release_pause:
                        last_emitted = None
                    if word and now - no_letter_since >= args.word_pause:
                        spoken = "".join(word)
                        print(f"\nWORD: {spoken}", flush=True)
                        speaker.speak(spoken)
                        word.clear()
                        last_emitted = None

                remaining = frame_period - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping.")
    except Exception as exc:
        print(f"Stream or inference error: {exc}")
        print("Check that the PC and ESP32-CAM share the same LAN and the URL ends in :81/stream.")
    finally:
        if word:
            spoken = "".join(word)
            print(f"\nWORD: {spoken}", flush=True)
            speaker.speak(spoken)
        speaker.close()


if __name__ == "__main__":
    main()
