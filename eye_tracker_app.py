import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from datetime import datetime
import time
import json
import threading
import queue




class EyeTracker:
   
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    LEFT_EYEBROW = [276, 283, 282, 295, 300]
    RIGHT_EYEBROW = [46, 53, 52, 65, 70]
    FOREHEAD = 10
   
    def __init__(self, smoothing_window=5, blink_threshold=0.21):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
       
        self.left_iris_history = deque(maxlen=smoothing_window)
        self.right_iris_history = deque(maxlen=smoothing_window)
        self.blink_threshold = blink_threshold
        self.eyes_closed = False
        self.blink_total = 0
        self.last_gaze = (0.5, 0.5)
       
    def _get_landmark_coords(self, landmarks, indices, frame_shape):
        h, w = frame_shape[:2]
        return np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices])
   
    def _get_iris_center(self, landmarks, iris_indices, frame_shape):
        coords = self._get_landmark_coords(landmarks, iris_indices, frame_shape)
        return np.mean(coords, axis=0).astype(int)
   
    def _calculate_ear(self, landmarks, eye_indices, frame_shape):
        coords = self._get_landmark_coords(landmarks, eye_indices, frame_shape)
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0
   
    def _get_eye_region(self, landmarks, eye_indices, frame_shape):
        coords = self._get_landmark_coords(landmarks, eye_indices, frame_shape)
        return (*coords.min(axis=0), *coords.max(axis=0))
   
    def _calculate_gaze_ratio(self, iris_center, eye_region):
        x_min, y_min, x_max, y_max = eye_region
        eye_width, eye_height = x_max - x_min, y_max - y_min
        if eye_width == 0 or eye_height == 0:
            return 0.5, 0.5
        return (np.clip((iris_center[0] - x_min) / eye_width, 0, 1),
                np.clip((iris_center[1] - y_min) / eye_height, 0, 1))
   
    def _smooth_coordinates(self, new_coord, history):
        history.append(new_coord)
        return np.mean(history, axis=0).astype(int) if history else new_coord
   
    def _determine_gaze_direction(self, x_ratio, y_ratio):
        h = 'destra' if x_ratio < 0.35 else 'sinistra' if x_ratio > 0.65 else 'centro'
        v = 'su' if y_ratio < 0.35 else 'giù' if y_ratio > 0.65 else 'centro'
        return h, v
   
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       
        data = {
            'timestamp': time.time(),
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'face_detected': False,
            'left_iris_x': None, 'left_iris_y': None,
            'right_iris_x': None, 'right_iris_y': None,
            'gaze_x': 0.5, 'gaze_y': 0.5,
            'gaze_horizontal': 'centro', 'gaze_vertical': 'centro',
            'left_ear': None, 'right_ear': None, 'avg_ear': None,
            'blink_detected': False, 'total_blinks': self.blink_total,
            'saccade_distance': 0.0,
        }
       
        if not results.multi_face_landmarks:
            return frame, data
       
        data['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
       
        # Iris positions
        left_iris = self._smooth_coordinates(
            self._get_iris_center(landmarks, self.LEFT_IRIS, frame.shape),
            self.left_iris_history)
        right_iris = self._smooth_coordinates(
            self._get_iris_center(landmarks, self.RIGHT_IRIS, frame.shape),
            self.right_iris_history)
       
        data['left_iris_x'], data['left_iris_y'] = int(left_iris[0]), int(left_iris[1])
        data['right_iris_x'], data['right_iris_y'] = int(right_iris[0]), int(right_iris[1])
       
        # EAR & Blink
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_EAR, frame.shape)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, frame.shape)
        avg_ear = (left_ear + right_ear) / 2
       
        data['left_ear'] = round(left_ear, 4)
        data['right_ear'] = round(right_ear, 4)
        data['avg_ear'] = round(avg_ear, 4)
       
        if avg_ear < self.blink_threshold:
            if not self.eyes_closed:
                self.eyes_closed = True
        else:
            if self.eyes_closed:
                self.eyes_closed = False
                self.blink_total += 1
                data['blink_detected'] = True
       
        data['total_blinks'] = self.blink_total
       
        # Gaze
        left_region = self._get_eye_region(landmarks, self.LEFT_EYE, frame.shape)
        right_region = self._get_eye_region(landmarks, self.RIGHT_EYE, frame.shape)
        left_ratio = self._calculate_gaze_ratio(left_iris, left_region)
        right_ratio = self._calculate_gaze_ratio(right_iris, right_region)
       
        avg_x = (left_ratio[0] + right_ratio[0]) / 2
        avg_y = (left_ratio[1] + right_ratio[1]) / 2
       
        data['gaze_x'] = round(avg_x, 4)
        data['gaze_y'] = round(avg_y, 4)
        data['gaze_horizontal'], data['gaze_vertical'] = self._determine_gaze_direction(avg_x, avg_y)
       
        # Saccade
        data['saccade_distance'] = round(np.sqrt((avg_x - self.last_gaze[0])**2 + (avg_y - self.last_gaze[1])**2), 4)
        self.last_gaze = (avg_x, avg_y)
       
        # Draw
        self._draw(frame, landmarks, left_iris, right_iris, data)
        return frame, data
   
    def _draw(self, frame, landmarks, left_iris, right_iris, data):
        h, w = frame.shape[:2]
       
        left_eye = self._get_landmark_coords(landmarks, self.LEFT_EYE, frame.shape)
        right_eye = self._get_landmark_coords(landmarks, self.RIGHT_EYE, frame.shape)
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.circle(frame, tuple(left_iris), 4, (255, 0, 255), -1)
        cv2.circle(frame, tuple(right_iris), 4, (255, 0, 255), -1)
       
        cv2.rectangle(frame, (10, 10), (280, 110), (40, 40, 40), -1)
        cv2.putText(frame, f"Sguardo: {data['gaze_horizontal']}, {data['gaze_vertical']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pos: ({data['gaze_x']:.2f}, {data['gaze_y']:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Blink: {data['total_blinks']} | EAR: {data['avg_ear']:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Saccade: {data['saccade_distance']:.4f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
       
        if data['blink_detected']:
            cv2.putText(frame, "BLINK!", (w//2 - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
   
    def reset(self):
        self.blink_total = 0
        self.last_gaze = (0.5, 0.5)




#  METRICS CALCULATOR


def calculate_metrics(data_list):
    """Calcola tutte le metriche dai dati raccolti."""
    if not data_list:
        return get_empty_metrics()
   
    df = pd.DataFrame(data_list)
    df = df[df['face_detected'] == True]
   
    if len(df) == 0:
        return get_empty_metrics()
   
    duration = df['relative_time'].max() if 'relative_time' in df else 0
    if duration == 0:
        return get_empty_metrics()
   
    # Blink
    total_blinks = df['blink_detected'].sum()
    bpm = (total_blinks / duration * 60) if duration > 0 else 0
   
    # Gaze variability
    gx_std = df['gaze_x'].std()
    gy_std = df['gaze_y'].std()
    variability = np.sqrt(gx_std**2 + gy_std**2)
   
    # Scan path
    scan_path = df['saccade_distance'].sum()
   
    # Direction counts
    h_counts = df['gaze_horizontal'].value_counts(normalize=True) * 100
    v_counts = df['gaze_vertical'].value_counts(normalize=True) * 100
   
    # Fixations (quando saccade < 0.02 per almeno 3 frame consecutivi)
    fixation_count = 0
    consecutive = 0
    for s in df['saccade_distance']:
        if s < 0.02:
            consecutive += 1
        else:
            if consecutive >= 3:
                fixation_count += 1
            consecutive = 0
   
    # Stress score
    stress = calculate_stress_score(bpm, variability, scan_path/duration if duration > 0 else 0,
                                    v_counts.get('giù', 0))
   
    return {
        'duration_seconds': round(duration, 2),
        'total_frames': len(df),
        'total_blinks': int(total_blinks),
        'blinks_per_minute': round(bpm, 2),
        'scan_path_length': round(scan_path, 4),
        'scan_path_velocity': round(scan_path / duration, 4) if duration > 0 else 0,
        'gaze_x_mean': round(df['gaze_x'].mean(), 4),
        'gaze_y_mean': round(df['gaze_y'].mean(), 4),
        'gaze_x_std': round(gx_std, 4),
        'gaze_y_std': round(gy_std, 4),
        'gaze_variability': round(variability, 4),
        'fixation_count': fixation_count,
        'avg_ear': round(df['avg_ear'].mean(), 4) if df['avg_ear'].notna().any() else 0,
        'pct_looking_left': round(h_counts.get('sinistra', 0), 1),
        'pct_looking_center': round(h_counts.get('centro', 0), 1),
        'pct_looking_right': round(h_counts.get('destra', 0), 1),
        'pct_looking_up': round(v_counts.get('su', 0), 1),
        'pct_looking_down': round(v_counts.get('giù', 0), 1),
        'stress_score': stress,
    }




def calculate_stress_score(bpm, variability, velocity, pct_down):
    score = 0
    if bpm > 30: score += 25
    elif bpm > 25: score += 15
    elif bpm < 10: score += 10
   
    if variability > 0.3: score += 25
    elif variability > 0.2: score += 15
   
    if velocity > 0.5: score += 20
    elif velocity > 0.3: score += 10
   
    if pct_down > 40: score += 20
    elif pct_down > 25: score += 10
   
    return min(100, score)




def get_empty_metrics():
    return {
        'duration_seconds': 0, 'total_frames': 0, 'total_blinks': 0,
        'blinks_per_minute': 0, 'scan_path_length': 0, 'scan_path_velocity': 0,
        'gaze_x_mean': 0.5, 'gaze_y_mean': 0.5, 'gaze_x_std': 0, 'gaze_y_std': 0,
        'gaze_variability': 0, 'fixation_count': 0, 'avg_ear': 0,
        'pct_looking_left': 0, 'pct_looking_center': 0, 'pct_looking_right': 0,
        'pct_looking_up': 0, 'pct_looking_down': 0, 'stress_score': 0,
    }




#  STREAMLIT APP


def main():
    st.set_page_config(page_title="Eye Tracker", page_icon="👁️", layout="wide")
   
    st.title("Eye Tracking - Analisi Stress")
   
    # Session state
    if 'data_list' not in st.session_state:
        st.session_state.data_list = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'session_name' not in st.session_state:
        st.session_state.session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   
    # Sidebar
    with st.sidebar:
        st.header("Impostazioni")
        blink_th = st.slider("Soglia Blink", 0.15, 0.30, 0.21, 0.01)
        camera_id = st.number_input("Camera ID", 0, 5, 0)
       
        st.divider()
        st.header("Sessione")
        st.session_state.session_name = st.text_input("Nome", st.session_state.session_name)
        st.metric("Frame salvati", len(st.session_state.data_list))
       
        if st.button("Reset Dati", use_container_width=True):
            st.session_state.data_list = []
            st.session_state.start_time = None
            st.rerun()
   
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Registra", "Metriche", "Esporta"])
   
    # TAB 1: RECORDING
    with tab1:
        st.subheader(" Registrazione Eye Tracking")
       
        st.info(" Clicca **Avvia Registrazione**, guarda la camera, poi clicca **Stop** quando hai finito.")
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            if st.button("Avvia Registrazione", use_container_width=True, type="primary"):
               
                # Reset data
                st.session_state.data_list = []
                st.session_state.start_time = time.time()
               
                tracker = EyeTracker(blink_threshold=blink_th)
                cap = cv2.VideoCapture(int(camera_id))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
               
                if not cap.isOpened():
                    st.error("Impossibile aprire la camera!")
                else:
                    video_placeholder = st.empty()
                    status_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                    stop_btn = st.button("STOP Registrazione", type="secondary")
                   
                    frame_count = 0
                   
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                       
                        processed_frame, data = tracker.process_frame(frame)
                       
                        # Add relative time and save
                        if data['face_detected']:
                            data['relative_time'] = round(time.time() - st.session_state.start_time, 3)
                            st.session_state.data_list.append(data)
                       
                        frame_count += 1
                       
                        # Update display every 2 frames
                        if frame_count % 2 == 0:
                            video_placeholder.image(
                                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_container_width=True
                            )
                           
                            elapsed = time.time() - st.session_state.start_time
                            status_placeholder.markdown(f"**REC** | ⏱️ {elapsed:.1f}s | 📊 {len(st.session_state.data_list)} frames")
                           
                            # Quick metrics
                            if len(st.session_state.data_list) > 0:
                                blinks = sum(1 for d in st.session_state.data_list if d['blink_detected'])
                                last = st.session_state.data_list[-1]
                                metrics_placeholder.markdown(
                                    f" Blink: **{blinks}** | "
                                    f" Sguardo: **{last['gaze_horizontal']}, {last['gaze_vertical']}** | "
                                    f" Pos: ({last['gaze_x']:.2f}, {last['gaze_y']:.2f})"
                                )
                       
                        # Check for stop (using keyboard in terminal: Ctrl+C)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                   
                    cap.release()
                    st.success(f" Registrazione completata! {len(st.session_state.data_list)} frame salvati.")
                    st.rerun()
       
        with col2:
            st.subheader(" Dati Attuali")
            if st.session_state.data_list:
                st.metric("Frame totali", len(st.session_state.data_list))
                metrics = calculate_metrics(st.session_state.data_list)
                st.metric("Durata (s)", metrics['duration_seconds'])
                st.metric("Blink totali", metrics['total_blinks'])
                st.metric("Stress Score", f"{metrics['stress_score']}/100")
            else:
                st.info("Nessun dato ancora")
   
    # TAB 2: METRICS
    with tab2:
        st.subheader(" Analisi Metriche")
       
        if not st.session_state.data_list:
            st.warning(" Nessun dato. Vai su 'Registra' per raccogliere dati.")
        else:
            metrics = calculate_metrics(st.session_state.data_list)
           
            # Metrics cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(" Durata", f"{metrics['duration_seconds']}s")
                st.metric(" Frame", metrics['total_frames'])
            with c2:
                st.metric(" Blink", metrics['total_blinks'])
                st.metric(" Blink/min", metrics['blinks_per_minute'])
            with c3:
                st.metric(" Scan Path", f"{metrics['scan_path_length']:.3f}")
                st.metric(" Fixations", metrics['fixation_count'])
            with c4:
                st.metric(" Stress", f"{metrics['stress_score']}/100")
                st.metric("⬇ % Giù", f"{metrics['pct_looking_down']}%")
           
            st.divider()
           
            # Charts
            df = pd.DataFrame(st.session_state.data_list)
           
            col1, col2 = st.columns(2)
           
            with col1:
                st.subheader(" Distribuzione Direzioni")
                dir_df = pd.DataFrame({
                    'Direzione': ['Sinistra', 'Centro', 'Destra', 'Su', 'Giù'],
                    'Percentuale': [metrics['pct_looking_left'], metrics['pct_looking_center'],
                                   metrics['pct_looking_right'], metrics['pct_looking_up'],
                                   metrics['pct_looking_down']]
                })
                fig = px.bar(dir_df, x='Direzione', y='Percentuale', color='Percentuale',
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
           
            with col2:
                st.subheader(" Heatmap Sguardo")
                if len(df) > 0:
                    fig = px.density_heatmap(df, x='gaze_x', y='gaze_y', nbinsx=15, nbinsy=15,
                                            color_continuous_scale='Hot')
                    fig.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])
                    st.plotly_chart(fig, use_container_width=True)
           
            # Timeline
            if 'relative_time' in df.columns and len(df) > 10:
                st.subheader("Timeline")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('Posizione Sguardo', 'EAR'))
                fig.add_trace(go.Scatter(x=df['relative_time'], y=df['gaze_x'], name='X', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['relative_time'], y=df['gaze_y'], name='Y', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['relative_time'], y=df['avg_ear'], name='EAR', line=dict(color='green')), row=2, col=1)
                fig.add_hline(y=0.21, line_dash="dash", line_color="red", row=2, col=1)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
   
    # TAB 3: EXPORT
    with tab3:
        st.subheader(" Esporta Dati")
       
        if not st.session_state.data_list:
            st.warning(" Nessun dato da esportare.")
        else:
            df = pd.DataFrame(st.session_state.data_list)
            metrics = calculate_metrics(st.session_state.data_list)
           
            st.success(f" **{len(df)} frame** pronti per l'export")
           
            with st.expander(" Anteprima dati (ultimi 20)"):
                st.dataframe(df.tail(20), use_container_width=True)
           
            # Download buttons
            c1, c2, c3 = st.columns(3)
           
            with c1:
                csv = df.to_csv(index=False)
                st.download_button(
                    " Scarica CSV",
                    csv,
                    f"{st.session_state.session_name}_data.csv",
                    "text/csv",
                    use_container_width=True
                )
           
            with c2:
                st.download_button(
                    " Scarica Metriche JSON",
                    json.dumps(metrics, indent=2),
                    f"{st.session_state.session_name}_metrics.json",
                    "application/json",
                    use_container_width=True
                )
           
            with c3:
                full = {
                    'session': st.session_state.session_name,
                    'recorded_at': datetime.now().isoformat(),
                    'metrics': metrics,
                    'data': df.to_dict('records')
                }
                st.download_button(
                    " Export Completo",
                    json.dumps(full, indent=2, default=str),
                    f"{st.session_state.session_name}_full.json",
                    "application/json",
                    use_container_width=True
                )
           
            st.divider()
           
            # Metrics summary table
            st.subheader(" Riepilogo Metriche")
            metrics_df = pd.DataFrame([
                {'Metrica': k, 'Valore': v} for k, v in metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)




if __name__ == "__main__":
    main()

