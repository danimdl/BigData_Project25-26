import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from tracker_engine import EyeTracker

# Shared file with Worker
WORKER_DATA_FILE = 'dashboard_data.json'

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eye Tracking Pro", layout="wide", page_icon="👁️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stMetric { background-color: #0e1117; border: 1px solid #303030; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def load_worker_data():
    """Reads processed data from Analytics Worker"""
    if os.path.exists(WORKER_DATA_FILE):
        try:
            with open(WORKER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_session_to_history():
    """Saves current session to local history"""
    if st.session_state.current_session_data:
        df = pd.DataFrame(st.session_state.current_session_data)
        
        # Final Stats
        duration = df['timestamp'].max() - df['timestamp'].min()
        avg_stress = df['stress_score'].mean() if 'stress_score' in df else 0
        total_blinks = df['total_blinks'].max() if 'total_blinks' in df else 0
        
        session_entry = {
            'id': len(st.session_state.sessions_archive) + 1,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': duration,
            'blinks': total_blinks,
            'avg_stress': avg_stress,
            'data': df
        }
        st.session_state.sessions_archive.append(session_entry)
        st.success(f"Session #{session_entry['id']} saved to archive!")

def main():
    st.title("👁️ Eye Tracking & Stress Analysis (Kafka Architecture)")

    # --- STATE INIT ---
    if 'recording' not in st.session_state: st.session_state.recording = False
    if 'tracker' not in st.session_state: st.session_state.tracker = EyeTracker()
    if 'sessions_archive' not in st.session_state: st.session_state.sessions_archive = []
    if 'current_session_data' not in st.session_state: st.session_state.current_session_data = []
    if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        cam_id = st.number_input("Webcam ID", 0, 5, 0)
        
        st.divider()
        blink_th = st.slider("Blink Threshold", 0.15, 0.35, 0.21, 0.01)
        st.session_state.tracker.blink_threshold = blink_th
        
        st.divider()
        silence_limit = st.slider("⏱️ Stop after silence (sec)", 2, 20, 5)

        st.divider()
        if st.button("🗑️ Reset Archive"):
            st.session_state.sessions_archive = []
            st.rerun()

    # --- MAIN UI ---
    tab1, tab2 = st.tabs(["🔴 Live Monitor", "📚 Session History"])

    with tab1:
        col_video, col_kpi = st.columns([2, 1])

        with col_video:
            # Control Buttons
            c1, c2 = st.columns(2)
            with c1:
                if not st.session_state.recording:
                    if st.button("▶️ START SESSION", type="primary", use_container_width=True):
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.tracker.session_id = st.session_state.session_id
                        st.session_state.tracker.reset_counters()
                        st.session_state.current_session_data = []
                        if os.path.exists(WORKER_DATA_FILE): os.remove(WORKER_DATA_FILE)
                        st.session_state.recording = True
                        st.rerun()
            with c2:
                if st.session_state.recording:
                    if st.button("⏹️ STOP", type="secondary", use_container_width=True):
                        st.session_state.recording = False
                        save_session_to_history()
                        st.rerun()

            video_placeholder = st.empty()
            warning_placeholder = st.empty()

        with col_kpi:
            st.markdown("### Real-time Metrics")
            kpi1, kpi2 = st.columns(2)
            metric_blink = kpi1.empty()
            metric_stress = kpi2.empty()
            
            st.divider()
            metric_status = st.empty()
            metric_gaze = st.empty()
            
            st.divider()
            st.markdown("##### Event Log (from Worker)")
            log_table = st.empty()

    # --- RECORDING LOOP ---
    if st.session_state.recording:
        cap = cv2.VideoCapture(int(cam_id))
        
        last_speech_time = time.time()

        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading webcam")
                break

            # 1. TRACKER PROCESSING
            frame, tracker_data = st.session_state.tracker.process_frame(frame)
            
            # Show Video
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 2. READ WORKER DATA
            worker_data = load_worker_data()
            
            stress_val = 0
            total_blinks = 0
            
            if worker_data:
                metrics = worker_data.get('metrics', {})
                total_blinks = metrics.get('total_blinks', 0)
                stress_val = metrics.get('stress_score', 0)
                is_talking = metrics.get('is_talking', False)
                gaze_dir = metrics.get('gaze_dir', "Center")
                
                # Update KPIs
                metric_blink.metric("Total Blinks", total_blinks)
                metric_stress.metric("Stress Score", f"{stress_val}/100", 
                                   delta=stress_val, delta_color="inverse")
                
                status_text = "🟢 SPEAKING" if is_talking else "🔴 SILENCE"
                metric_status.info(f"Voice Status: {status_text}")
                metric_gaze.success(f"Gaze Direction: **{gaze_dir}**")
                
                # Update Table
                if 'table_data' in worker_data:
                    log_table.dataframe(worker_data['table_data'], height=200, use_container_width=True)

                # 3. AUTO-STOP LOGIC
                if is_talking:
                    last_speech_time = time.time()
                else:
                    elapsed = time.time() - last_speech_time
                    if elapsed > 1:
                        warning_placeholder.warning(f"⚠️ Silence detected: {elapsed:.1f}s / {silence_limit}s")
                    
                    if elapsed > silence_limit:
                        st.toast("🛑 Auto-stop due to silence!", icon="🤐")
                        st.session_state.recording = False
                        tracker_data['stress_score'] = stress_val
                        tracker_data['total_blinks'] = total_blinks
                        st.session_state.current_session_data.append(tracker_data)
                        save_session_to_history()
                        cap.release()
                        st.rerun()
                        break
            
            # Data Enrichment for History
            tracker_data['stress_score'] = stress_val
            tracker_data['total_blinks'] = total_blinks
            st.session_state.current_session_data.append(tracker_data)

            time.sleep(0.01)

        cap.release()

    # --- HISTORY TAB ---
    with tab2:
        if st.session_state.sessions_archive:
            for s in reversed(st.session_state.sessions_archive):
                with st.expander(f"Session #{s['id']} - {s['date']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Duration", f"{s['duration']:.1f}s")
                    c2.metric("Total Blinks", s['blinks'])
                    c3.metric("Avg Stress", f"{s['avg_stress']:.1f}")
                    
                    st.line_chart(s['data']['stress_score'])
                    st.dataframe(s['data'].head(50))
        else:
            st.info("No sessions recorded yet.")

if __name__ == "__main__":
    main()