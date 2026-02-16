# BigData_Project25-26: Eye Tracking & Stress Analysis System

An advanced real-time eye-tracking system designed for stress analysis and behavioral monitoring. The system utilizes MediaPipe Face Mesh for high-precision biometric tracking and Apache Kafka to support high-throughput, distributed big data analytics.

The architecture follows a microservices-oriented pattern where the Tracker (Producer) generates raw data, the Analytics Worker (Consumer) processes metrics in real-time, and the Streamlit Dashboard visualizes the insights.

## Key Features

- **High-Precision Eye Tracking:** Utilizes 478 facial landmarks via MediaPipe with sub-pixel accuracy.
- **Spiderweb Mesh Visualization:** Custom visual feedback connecting the iris to key eye landmarks for precise tracking verification.

### Real-Time Analytics

- **Blink Detection:** Based on Eye Aspect Ratio (EAR) with dynamic thresholding.
- **Speech Detection:** Mouth Aspect Ratio (MAR) analysis to detect speaking status.
- **Gaze Direction:** Classification of gaze into 5 zones (Center, Left, Right, Up, Down).
- **Stress Scoring Engine:** A proprietary algorithm (0-100) combining Blink Rate, Gaze Variability, Scan Velocity, and Downward Gaze percentage.

### Big Data Architecture

- **Kafka Producer:** Streams raw biometric data at high frequency (~30Hz).
- **Kafka Consumer (Worker):** Processes data, calculates aggregates, and handles file I/O safely (Windows-compatible).
- **Aggregation Topic:** Sends periodic summaries (every 5s) to a dedicated Kafka topic for long-term storage.
- **Interactive Dashboard:** A Streamlit-based UI with real-time charts, event logs, and session history.


## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Biometrics | MediaPipe Face Landmarker (Refined Mesh Model) | Face Landmark Detection |
| Computer Vision | OpenCV | Frame acquisition and visual overlays |
| Streaming | Apache Kafka | Message broker for real-time data pipelines |
| Data Processing | Python (Pandas/NumPy) | Metric calculation and statistical analysis |
| Frontend | Streamlit | Real-time monitoring dashboard and history viewer |
| Containerization | Docker | Kafka & Zookeeper orchestration |

## Requirements
-confluent-kafka
-streamlit>=1.28.0
-opencv-python-headless>=4.8.0
-mediapipe==0.10.11
-numpy>=1.24.0,<2.0.0
-pandas>=2.0.0
-plotly>=5.18.0
- Python: 3.10 or 3.11 (Recommended)
- Docker Desktop: Required for running the Kafka environment.
- Webcam: Functional camera for eye tracking.

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/BigData_Project25-26.git
cd BigData_Project25-26
```

### Set up the Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

To start the full system, you need to run three separate components in this specific order:

### 1. Start the Infrastructure (Kafka)

Ensure Docker Desktop is running, then execute:

```bash
docker-compose up -d
```

Wait about 30 seconds for the Broker to fully initialize.

### 2. Start the Analytics Brain (Worker)

Open a terminal and run the worker. This service listens to raw data, calculates stress, and writes to the dashboard files.

```bash
python analytics_worker.py
```

You should see a message indicating the Worker Analytics is online.

### 3. Start the User Interface (Streamlit)

Open a new terminal and launch the dashboard. This will also start the Tracker engine when you press "Start".

```bash
streamlit run app.py
```

## Methodology & Algorithms

### 1. Eye Aspect Ratio (EAR)

Used for blink detection. It measures the openness of the eye.

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

A blink is registered when EAR falls below the user-defined threshold (default: 0.21).

### 2. Envelope Data Pattern

Data is streamed to Kafka using a structured "Envelope" format to support schema evolution:

```json
{
  "timestamp": 1771166790.88,
  "type": "REALTIME_DATA",
  "metrics": {
      "gaze_x": 0.395,
      "gaze_y": 0.703,
      "total_blinks": 26,
      "stress_score": 45
  }
}


### 3. Stress Score Calculation (0-100)

The stress level is a composite metric calculated in `analytics_utils.py`:

- **Blink Rate (30%):** Deviations from normal resting rate (12-20 bpm). High rates indicate anxiety; extremely low rates indicate cognitive overload.
- **Gaze Variability (30%):** Standard deviation of X/Y coordinates. Erratic movement increases score.
- **Scan Velocity (25%):** Speed of saccadic movements.
- **Downward Gaze (15%):** Percentage of time looking down (associated with negative affect or avoidance).

## Project Structure

- `app.py`: The Frontend. Displays the video feed, charts, and handles user session controls.
- `tracker_engine.py`: The Producer. Runs the MediaPipe pipeline and sends raw data to Kafka (gaze_data).
- `analytics_worker.py`: The Consumer. Processes raw data, aggregates metrics, and updates the dashboard JSON.
- `analytics_utils.py`: The Logic. Contains the mathematical formulas for stress and metric calculation.
- `docker-compose.yml`: Configuration for Kafka and Zookeeper containers.

## Troubleshooting

### WinError 5 / Access Denied

This is handled automatically by the Worker's "Retry Logic". If it persists, ensure no other program has `dashboard_data.json` open.

### Kafka Connection Error

- Check if Docker containers are running: `docker ps`
- Ensure port 9092 is not blocked by a firewall.

### AttributeError in App

Ensure you are running the latest version of `app.py` and that the Worker is running to generate the required JSON files.

## Authors

Developed for the Big Data Course 2025-26 - University of Camerino (UNICAM).

- Daniela Maria Di Lucchio
- Marco Francoletti
- Lorenzo Marcantognini

##  Bibliography
-Kafka documentation: https://kafka.apache.org/41/getting-started/introduction/
-Streamlit documentation: https://docs.streamlit.io/
-Open CV documentation http://opencv.org/
-Docker documentation: https://docs.docker.com/manuals/