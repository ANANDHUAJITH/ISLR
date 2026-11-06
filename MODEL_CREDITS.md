# Model provenance

- `models/islr_mediapipe_landmarks_svm.joblib` is the project classifier trained from the `islr_data_sample` folder structure in [ANANDHUAJITH/ISLR](https://github.com/ANANDHUAJITH/ISLR/tree/main/islr_data_sample). The training code is `host/train_classifier.py`; metrics are in `reports/`.
- `models/hand_landmarker.task` was extracted from the `islr.task` bundle in the same user-provided repository. It is used only for pretrained hand detection/landmark extraction; this project does not train or modify the hand-landmarker network.

Before redistributing these model binaries publicly, review the upstream repository and model-asset licensing/attribution requirements. This prototype includes no copy of the original image dataset.
