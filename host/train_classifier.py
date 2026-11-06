#!/usr/bin/env python3
"""Train an ISL alphabet classifier on MediaPipe HandLandmarker features.

Extracts 21 hand landmarks (x, y, z), normalizes them around the wrist, then
fits a separate SVM. This is not MediaPipe Model Maker and does not export a
MediaPipe GestureRecognizer .task file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def landmark_feature(result) -> np.ndarray:
    """Return a wrist-centered, scale-normalized 64-value descriptor."""
    if not result.hand_landmarks:
        return np.zeros(64, dtype=np.float32)  # last value marks no detected hand
    index = 0
    if result.handedness and len(result.handedness) > 0:
        index = max(range(len(result.handedness)), key=lambda i: result.handedness[i][0].score)
    hand = result.hand_landmarks[index]
    points = np.asarray([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
    points -= points[0].copy()  # wrist anchor
    handedness = None
    if result.handedness and len(result.handedness) > index:
        handedness = result.handedness[index][0].category_name.lower()
    if handedness == "left":
        points[:, 0] *= -1.0  # canonicalize mirrored hands to one convention
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale > 1e-6:
        points /= scale
    return np.concatenate([points.reshape(-1), np.asarray([1.0], dtype=np.float32)])


def load_features(dataset_root: Path, hand_model: Path):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not hand_model.is_file():
        raise FileNotFoundError(f"MediaPipe hand model not found: {hand_model}")
    paths, labels = [], []
    counts = {}
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
        counts[class_dir.name] = len(images)
        for path in images:
            paths.append(path)
            labels.append(class_dir.name)
    if not paths or min(counts.values(), default=0) < 2:
        raise ValueError("Need at least two supported images in every label folder")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
    features, detected = [], 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for i, path in enumerate(paths, 1):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            detected += bool(result.hand_landmarks)
            features.append(landmark_feature(result))
            if i % 100 == 0 or i == len(paths):
                print(f"Extracted landmarks: {i}/{len(paths)} images")
    detection_rate = detected / len(paths)
    print(f"MediaPipe detected a hand in {detected}/{len(paths)} images ({detection_rate:.1%})")
    print(f"Loaded {len(paths)} images across {len(counts)} classes")
    print("Class counts:", json.dumps(counts, sort_keys=True))
    return np.stack(features), np.asarray(labels), counts, detection_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing one folder per label")
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "hand_landmarker.task")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts_mediapipe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y, class_counts, detection_rate = load_features(args.data, args.hand_model)
    classes = sorted(np.unique(y).tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", SVC(kernel="rbf", class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    search = GridSearchCV(
        pipeline,
        param_grid={"classifier__C": [1, 10, 100, 1000], "classifier__gamma": ["scale", "auto"]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=2,
        refit=True,
        verbose=1,
    )
    print(f"Training split: {len(y_train)} images; held-out test split: {len(y_test)} images")
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    report_text = classification_report(y_test, predictions, labels=classes, zero_division=0)
    report_json = classification_report(y_test, predictions, labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=classes)

    print("\nBest CV macro-F1:", round(float(search.best_score_), 4))
    print("Best parameters:", search.best_params_)
    print(f"Held-out accuracy: {accuracy:.4f}")
    print(f"Held-out macro-F1: {macro_f1:.4f}")
    print("\nPer-class held-out report:\n" + report_text)

    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", SVC(kernel="rbf", class_weight="balanced")),
    ])
    final_model.set_params(**search.best_params_)
    final_model.fit(X, y)
    model_path = args.output_dir / "islr_mediapipe_landmarks_svm.joblib"
    joblib.dump({
        "model": final_model,
        "classes": classes,
        "feature_description": "MediaPipe 21 hand landmarks; wrist-centered, scale-normalized xyz plus hand-present flag",
        "class_counts": class_counts,
        "hand_model_path_at_training": str(args.hand_model.resolve()),
        "mediapipe_version": mp.__version__,
    }, model_path)

    metrics = {
        "dataset": str(args.data.resolve()),
        "total_images": int(len(y)),
        "class_counts": class_counts,
        "labels": classes,
        "mediapipe_hand_detection_rate": detection_rate,
        "holdout_fraction": 0.25,
        "random_seed": args.seed,
        "cv_best_macro_f1": float(search.best_score_),
        "best_params": search.best_params_,
        "heldout_accuracy": accuracy,
        "heldout_macro_f1": macro_f1,
        "heldout_report": report_json,
        "model_type": "MediaPipe Hands landmarks + scikit-learn SVM (not a .task export)",
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d", header=",".join(classes), comments="")
    print(f"\nSaved model: {model_path}")
    print(f"Saved evaluation: {args.output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
