# BigData_Project25-26: Eye Tracking - Stress Analysis

Real-time eye tracking system for stress analysis using MediaPipe Face Mesh.

## Requirements

- Python 3.11 (MediaPipe is not compatible with Python 3.13)
- Webcam

## Installation

### 1. Install Python 3.11

Download from: https://www.python.org/downloads/release/python-3119/

During installation, check "Add Python to PATH".

### 2. Create virtual environment

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit opencv-python mediapipe numpy pandas plotly
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run eye_tracker_app.py
```

The app will open in your browser at `http://localhost:8501`

**Important:** Do NOT run with `python eye_tracker_app.py` - it won't work.

## How to use

1. Go to **"Registra"** tab
2. Click **"Avvia Registrazione"**
3. Look at the camera
4. Press **Ctrl+C** in terminal to stop recording
5. Go to other tabs to see analysis and export data

## Metrics

| Metric | Description |
|--------|-------------|
| **Gaze X, Y** | Eye position (0-1 normalized coordinates) |
| **Gaze Direction** | left/center/right, up/center/down |
| **EAR** | Eye Aspect Ratio - measures eye openness |
| **Blink Detection** | Detected when EAR < 0.21 |
| **Blink Rate** | Blinks per minute |
| **Saccade Distance** | Eye movement between frames |
| **Fixation Count** | Number of gaze fixations |
| **Head Pose** | Pitch (up/down), Yaw (left/right), Roll (tilt) |
| **Stress Score** | 0-100 derived from all metrics |

## Project Structure

```
eye-tracking/
├── eye_tracker_app.py    # Main application
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **MediaPipe** | Google framework for face landmark detection (478 points) |
| **OpenCV** | Video capture and image processing |
| **NumPy** | Mathematical calculations (EAR, distances, etc.) |
| **Pandas** | Data management and CSV export |
| **Plotly** | Interactive charts (heatmap, timeline, scanpath) |
| **Streamlit** | Web UI |

## Export formats

- **CSV** - All frame-by-frame data
- **JSON** - Metrics summary
- **JSON Full** - Complete export (metrics + data)

## Troubleshooting

### "MediaPipe not found" or installation errors
Make sure you're using Python 3.11, not 3.12 or 3.13.

### Camera not opening
- Check Camera ID in sidebar (try 0, 1, 2)
- Make sure no other app is using the webcam

### Streamlit warnings about ScriptRunContext
You're running with `python` instead of `streamlit run`. Use:
```bash
streamlit run eye_tracker_app.py
```

## References

- MediaPipe Face Landmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
