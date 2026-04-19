## Testing in MediaPipe Studio
 
[MediaPipe Studio](https://mediapipe-studio.webapps.google.com/) lets you test `.task` files directly in your browser — **no code needed**.
 
### Steps
 
1. **Open MediaPipe Studio**
   Go to [https://mediapipe-studio.webapps.google.com/studio/demo/hand_landmarker](https://mediapipe-studio.webapps.google.com/studio/demo/hand_landmarker)
2. **Upload the model**
   - In the left panel, click **"Change model"**
   - Select **"Upload custom model"**
   - Upload your `islr.task` file
3. **Choose your input**
   - **Camera** — use your webcam for live testing
   - **Image** — upload a still photo of a hand sign
   - **Video** — upload a recorded clip
4. **Adjust inference settings** (optional)
   | Parameter | Recommended |
   |---|---|
   | Number of Hands | `1` or `2` |
   | Min Hand Detection Confidence | `0.5` |
   | Min Hand Presence Confidence | `0.5` |
   | Min Tracking Confidence | `0.5` |
5. **Observe the output**
   - 21 landmark points overlaid on the hand
   - X, Y, Z coordinates for each joint
   - Handedness (Left / Right)
> **Tip:** For ISL testing, ensure your hand is well-lit and the background is clean. The model was optimized for single-hand frontal gestures.
 
---
