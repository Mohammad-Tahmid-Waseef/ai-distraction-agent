# AI Distraction Agent — Focus Tracker (Face Pose + Live Stats)

AI Distraction Agent is a real-time focus tracker that uses your webcam and **MediaPipe’s pretrained face landmark model** (ML) to estimate head pose (pitch/yaw/roll). A lightweight rule-based layer then classifies whether you’re **Focused** or **Distracted**, while the app tracks session stats, logs events, and graphs focus over time.

---

## ✨ Features

- ✅ **Real-time webcam tracking** (OpenCV)
- ✅ **Face landmark detection (468 points)** with MediaPipe Face Mesh
- ✅ **Head pose estimation** → pitch/yaw/roll (solvePnP + Euler angles)
- ✅ **Focus classification** using simple thresholds:
  - Looking left/right = distracted (yaw threshold)
  - Looking up/down = distracted (pitch threshold)
  - No face detected = distracted (you left / camera can’t see you)
- ✅ **Live session statistics**
  - Focus time vs distracted time
  - Focus percentage
- ✅ **Focus-over-time graph** (last 30 minutes)
- ✅ **Event log** with timestamps (state changes + actions)
- ✅ **Export session data to JSON** (`focus_log_YYYYMMDD_HHMMSS.json`)
- ✅ **Threaded video processing** so the GUI stays responsive

---

## 🧠 How it works (high level)

1. Webcam frame is captured with OpenCV  
2. MediaPipe detects face mesh landmarks  
3. A small set of landmark points are used to estimate head pose:
   - `cv2.solvePnP()` → rotation/translation vectors  
   - Convert to Euler angles (pitch/yaw/roll)
4. Simple threshold rules classify **Focused** vs **Distracted**
5. GUI updates stats + graph + logs events

---

## 📦 Tech Stack

- **Python**
- **OpenCV** (camera + solvePnP)
- **MediaPipe** (Face Mesh landmarks)
- **Tkinter** (desktop UI)
- **Matplotlib** (embedded live graph)
- **Pillow (PIL)** (image display inside Tkinter)

---

## ✅ Requirements

- Python **3.8+** (recommended 3.10+)
- A working webcam
- Decent lighting (face should be visible)

---

## 🚀 Installation & Run

From inside the project folder:

```bash
pip install -r requirements.txt

Then type: 
python main.py


