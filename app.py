import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from tracker_engine import EyeTracker
from analytics_utils import (
    calculate_metrics, 
    add_relative_time, 
    calculate_saccade_distance,
    get_stress_level
)

WORKER_DATA_FILE = 'dashboard_data.json'

# ========== NUMPY/PANDAS JSON ENCODER ==========
class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON personalizzato per gestire tipi numpy/pandas"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return 0
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return 0
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

# --- FUNZIONI DI UTILITÀ ---
def calculate_advanced_metrics(df):
    """Calcola le metriche per il report finale"""
    if df.empty: 
        return {
            'duration': 0,
            'scan_path': 0,
            'fixations': 0,
            'pct_talking': 0,
            'stress_avg': 0
        }
    
    duration = df['timestamp'].max() - df['timestamp'].min()
    if duration <= 0:
        duration = 0.1
    
    scan_path = 0
    if 'gaze_x' in df.columns:
        df['dist'] = np.sqrt(df['gaze_x'].diff()**2 + df['gaze_y'].diff()**2)
        scan_path = df['dist'].sum() * 100
    
    fixations = (df['saccade_distance'] < 0.01).sum() if 'saccade_distance' in df.columns else 0
    talking_frames = df['is_talking'].sum() if 'is_talking' in df.columns else 0
    pct_talking = (talking_frames / len(df)) * 100 if len(df) > 0 else 0
    
    if 'stress_score' in df.columns:
        stress_avg = df['stress_score'].mean()
        if np.isnan(stress_avg) or np.isinf(stress_avg):
            stress_avg = 0
    else:
        stress_avg = 0
    
    return {
        'duration': round(duration, 2),
        'scan_path': round(scan_path, 2),
        'fixations': int(fixations),
        'pct_talking': round(pct_talking, 1),
        'stress_avg': round(stress_avg, 1)
    }

def load_live_data():
    """Carica dati live dal worker (opzionale)"""
    if os.path.exists(WORKER_DATA_FILE):
        try:
            with open(WORKER_DATA_FILE, 'r') as f:
                return json.load(f)
        except: 
            pass
    return None

def save_and_stop():
    """
    Funzione unica per salvare e fermare (chiamata da bottone o timer).
    CALCOLA LO STRESS PER OGNI FRAME usando finestre scorrevoli.
    """
    st.session_state.recording = False
    
    if st.session_state.current_session_data:
        # 1. Prepara i dati completi
        data_copy = st.session_state.current_session_data.copy()
        data_with_time = add_relative_time(data_copy)
        data_with_saccades = calculate_saccade_distance(data_with_time)
        
        # 2. Calcola stress in finestre scorrevoli
        window_size = 50  # Finestra mobile di 50 frame
        
        for i in range(len(data_with_saccades)):
            start_idx = max(0, i - window_size + 1)
            window_data = data_with_saccades[start_idx:i+1]
            
            if len(window_data) >= 10:
                try:
                    window_metrics = calculate_metrics(window_data)
                    data_with_saccades[i]['stress_score'] = window_metrics['stress_score']
                    data_with_saccades[i]['stress_level'] = window_metrics['stress_level']
                except:
                    data_with_saccades[i]['stress_score'] = 0
                    data_with_saccades[i]['stress_level'] = 'basso'
            else:
                data_with_saccades[i]['stress_score'] = 0
                data_with_saccades[i]['stress_level'] = 'basso'
        
        # 3. Converti in DataFrame
        df_session = pd.DataFrame(data_with_saccades)
        
        # 4. Calcola metriche avanzate
        metrics = calculate_advanced_metrics(df_session)
        
        # 5. Salva sessione
        session_id = len(st.session_state.sessions_archive) + 1
        archive_entry = {
            'id': session_id,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': metrics,
            'full_data': df_session
        }
        st.session_state.sessions_archive.append(archive_entry)
        return True
    return False

# ========== NUOVE FUNZIONI ==========

def calculate_realtime_metrics(data_buffer):
    """Calcola metriche in tempo reale"""
    if len(data_buffer) < 10:
        return None
    
    try:
        data_copy = data_buffer.copy()
        data_with_time = add_relative_time(data_copy)
        data_with_saccades = calculate_saccade_distance(data_with_time)
        metrics = calculate_metrics(data_with_saccades)
        return metrics
    except Exception as e:
        print(f"Errore calcolo metriche: {e}")
        return None

def create_gaze_heatmap(df):
    """Crea una heatmap 2D dello sguardo"""
    if df.empty or 'gaze_x' not in df.columns:
        return None
    
    try:
        fig = go.Figure(data=go.Histogram2d(
            x=df['gaze_x'],
            y=df['gaze_y'],
            colorscale='Hot',
            nbinsx=20,
            nbinsy=20
        ))
        fig.update_layout(
            title="Heatmap Gaze",
            xaxis_title="Gaze X",
            yaxis_title="Gaze Y",
            height=400
        )
        return fig
    except:
        return None

def create_stress_timeline(df):
    """Crea un grafico timeline dello stress colorato"""
    if df.empty or 'stress_score' not in df.columns:
        return None
    
    try:
        df['stress_color'] = df['stress_score'].apply(
            lambda x: 'green' if x < 30 else 'orange' if x < 60 else 'red'
        )
        
        fig = px.line(
            df.reset_index(), 
            x='index', 
            y='stress_score',
            title="Stress Timeline",
            color='stress_color',
            color_discrete_map={'green': 'green', 'orange': 'orange', 'red': 'red'}
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
    except:
        return None

def create_blink_timeline(df):
    """Crea timeline dei blink"""
    if df.empty or 'blink_detected' not in df.columns:
        return None
    
    try:
        blink_points = df[df['blink_detected'] == True].reset_index()
        if len(blink_points) > 0:
            fig = px.scatter(
                blink_points,
                x='index',
                y=[1] * len(blink_points),
                title=f"Blink Timeline (Totale: {len(blink_points)})",
                labels={'index': 'Frame', 'y': ''}
            )
            fig.update_layout(height=150, showlegend=False)
            fig.update_yaxis(visible=False)
            return fig
    except:
        pass
    return None

def get_aggregate_stats(sessions_archive):
    """Calcola statistiche aggregate su tutte le sessioni"""
    if not sessions_archive:
        return None
    
    total_duration = sum(s['summary']['duration'] for s in sessions_archive)
    total_sessions = len(sessions_archive)
    
    stress_values = [s['summary']['stress_avg'] for s in sessions_archive if not np.isnan(s['summary']['stress_avg'])]
    talking_values = [s['summary']['pct_talking'] for s in sessions_archive if not np.isnan(s['summary']['pct_talking'])]
    
    avg_stress = np.mean(stress_values) if stress_values else 0
    avg_talking = np.mean(talking_values) if talking_values else 0
    
    return {
        'total_sessions': total_sessions,
        'total_duration': round(total_duration, 2),
        'avg_stress': round(avg_stress, 1),
        'avg_talking': round(avg_talking, 1)
    }

def export_session_json(session):
    """Esporta sessione completa in JSON"""
    export_data = {
        'id': session['id'],
        'date': session['date'],
        'summary': session['summary'],
        'data': session['full_data'].to_dict('records')
    }
    return json.dumps(export_data, indent=2, cls=NumpyEncoder)

# ========== MAIN APPLICATION ==========

def main():
    st.set_page_config(page_title="Eye Tracking Auto-Stop", layout="wide")
    st.title("👁️ Eye Tracking - Auto Stop System")

    # INIZIALIZZAZIONE
    if 'recording' not in st.session_state: 
        st.session_state.recording = False
    if 'tracker' not in st.session_state: 
        st.session_state.tracker = EyeTracker()
    if 'sessions_archive' not in st.session_state: 
        st.session_state.sessions_archive = []
    if 'current_session_data' not in st.session_state: 
        st.session_state.current_session_data = []
    if 'last_metrics' not in st.session_state: 
        st.session_state.last_metrics = None

    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configurazione")
        cam_id = st.number_input("Camera ID", 0, 5, 0)
        blink_th = st.slider("Soglia Blink (EAR)", 0.15, 0.35, 0.21, 0.01)
        mouth_th = st.slider("Soglia Parlato (MAR)", 0.005, 0.020, 0.012, 0.001)
        
        st.session_state.tracker.blink_threshold = blink_th
        st.session_state.tracker.mouth_threshold = mouth_th
        
        st.divider()
        silence_limit = st.slider("⏱️ Stop dopo silenzio (sec)", 2, 10, 5)

        st.divider()
        st.subheader("📈 Statistiche Totali")
        agg_stats = get_aggregate_stats(st.session_state.sessions_archive)
        if agg_stats:
            st.metric("Sessioni Totali", agg_stats['total_sessions'])
            st.metric("Durata Totale", f"{agg_stats['total_duration']:.0f}s")
            st.metric("Stress Medio", f"{agg_stats['avg_stress']:.1f}/100")
            st.metric("Parlato Medio", f"{agg_stats['avg_talking']:.1f}%")
        else:
            st.info("Nessuna sessione registrata")
        
        st.divider()
        if st.button("🗑️ Reset Archivio"):
            st.session_state.sessions_archive = []
            st.rerun()

    # TABS
    tab_live, tab_metrics, tab_history = st.tabs(["🔴 Live & Auto-Stop", "📊 Report", "📚 Storico"])

    # TAB 1: LIVE
    with tab_live:
        col_video, col_info = st.columns([1.5, 1])
        
        with col_video:
            c1, c2 = st.columns(2)
            with c1:
                if not st.session_state.recording:
                    if st.button("▶️ START SESSIONE", type="primary", use_container_width=True):
                        st.session_state.current_session_data = [] 
                        st.session_state.tracker.reset_counters()
                        if os.path.exists(WORKER_DATA_FILE): 
                            os.remove(WORKER_DATA_FILE)
                        st.session_state.recording = True
                        st.rerun()
            with c2:
                if st.session_state.recording:
                    if st.button("⏹️ STOP MANUALE", type="secondary", use_container_width=True):
                        save_and_stop()
                        st.rerun()

            video_place = st.empty()
            silence_warning = st.empty()

        with col_info:
            st.markdown("#### Real-time KPI")
            k1, k2 = st.columns(2)
            p_blink = k1.empty()
            p_stress = k2.empty()
            
            p_blink.metric("Blink", 0)
            p_stress.metric("Stress", "0/100")
            
            st.divider()
            k3, k4 = st.columns(2)
            p_bpm = k3.empty()
            p_ear = k4.empty()
            p_bpm.metric("Blink/min", "0")
            p_ear.metric("Avg EAR", "0.00")
            
            st.divider()
            p_voice = st.empty()
            p_timer = st.empty()
            
            st.divider()
            st.markdown("#### 📉 Stress Live (ultimi 100 frame)")
            stress_chart_placeholder = st.empty()

    # LOOP REGISTRAZIONE
    if st.session_state.recording:
        try:
            cap = cv2.VideoCapture(int(cam_id))
            if not cap.isOpened():
                st.error(f"❌ Impossibile aprire camera {cam_id}")
                st.session_state.recording = False
                st.stop()
        except Exception as e:
            st.error(f"❌ Errore inizializzazione camera: {e}")
            st.session_state.recording = False
            st.stop()
        
        last_speech_time = time.time()
        
        while st.session_state.recording:
            try:
                ret, frame = cap.read()
                if not ret: 
                    st.warning("⚠️ Frame non disponibile")
                    break
                
                frame, data_point = st.session_state.tracker.process_frame(frame)
                video_place.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                is_talking_now = data_point['is_talking']
                
                if is_talking_now:
                    last_speech_time = time.time()
                    silence_warning.success("🎙️ Voce rilevata - Timer resettato")
                else:
                    elapsed_silence = time.time() - last_speech_time
                    prog_val = min(elapsed_silence / silence_limit, 1.0)
                    
                    if elapsed_silence > 1.0:
                        silence_warning.progress(
                            prog_val, 
                            text=f"⚠️ Silenzio: {elapsed_silence:.1f}s / {silence_limit}s"
                        )
                    
                    if elapsed_silence > silence_limit:
                        st.toast(f"🛑 Sessione terminata per silenzio (> {silence_limit}s)!", icon="🤐")
                        save_and_stop()
                        st.rerun()
                        break

                st.session_state.current_session_data.append(data_point)
                
                if len(st.session_state.current_session_data) % 10 == 0:
                    metrics = calculate_realtime_metrics(st.session_state.current_session_data)
                    
                    if metrics:
                        st.session_state.last_metrics = metrics
                        
                        st.session_state.tracker.update_stress_display(
                            metrics['stress_score'],
                            metrics['stress_level']
                        )
                        
                        p_blink.metric("Blink", metrics['total_blinks'])
                        p_stress.metric(
                            "Stress", 
                            f"{metrics['stress_score']}/100",
                            delta=f"{metrics['stress_level']}",
                            delta_color="off"
                        )
                        p_bpm.metric("Blink/min", f"{metrics['blinks_per_minute']:.1f}")
                        p_ear.metric("Avg EAR", f"{metrics['avg_ear']:.3f}")
                        
                        df_temp = pd.DataFrame(st.session_state.current_session_data[-100:])
                        if 'stress_score' in df_temp.columns:
                            stress_chart_placeholder.line_chart(df_temp['stress_score'])

                vt = "🟢 PARLANDO" if is_talking_now else "🔴 SILENZIO"
                p_voice.metric("Stato Voce", vt)
                p_timer.metric("Ultima voce", f"{time.time() - last_speech_time:.1f}s fa")
                
                time.sleep(0.01)
                
            except Exception as e:
                st.error(f"⚠️ Errore nel loop: {e}")
                time.sleep(0.1)
                continue
                
        cap.release()

    # TAB 2: METRICHE
    with tab_metrics:
        if st.session_state.sessions_archive:
            last_session = st.session_state.sessions_archive[-1]
            m = last_session['summary']
            st.success(f"✅ Ultima Sessione (#{last_session['id']}) - {last_session['date']}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Durata", f"{m['duration']:.1f}s")
            c2.metric("% Parlato", f"{m['pct_talking']:.1f}%")
            c3.metric("Fixations", m['fixations'])
            c4.metric("Stress Medio", f"{m['stress_avg']:.1f}/100")
            
            df = last_session['full_data']
            if not df.empty:
                st.divider()
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    stress_fig = create_stress_timeline(df)
                    if stress_fig:
                        st.plotly_chart(stress_fig, use_container_width=True)
                    
                    blink_fig = create_blink_timeline(df)
                    if blink_fig:
                        st.plotly_chart(blink_fig, use_container_width=True)
                
                with col_g2:
                    heatmap_fig = create_gaze_heatmap(df)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                
                st.divider()
                st.subheader("📋 Statistiche Dettagliate")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Gaze X medio", f"{df['gaze_x'].mean():.3f}")
                    st.metric("Gaze Y medio", f"{df['gaze_y'].mean():.3f}")
                
                with col_s2:
                    h_counts = df['gaze_horizontal'].value_counts(normalize=True) * 100
                    st.metric("% Sinistra", f"{h_counts.get('sinistra', 0):.1f}%")
                    st.metric("% Destra", f"{h_counts.get('destra', 0):.1f}%")
                
                with col_s3:
                    v_counts = df['gaze_vertical'].value_counts(normalize=True) * 100
                    st.metric("% Su", f"{v_counts.get('su', 0):.1f}%")
                    st.metric("% Giù", f"{v_counts.get('giu', 0):.1f}%")
                
                st.divider()
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Scarica CSV",
                        csv,
                        f"session_{last_session['id']}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col_e2:
                    json_data = export_session_json(last_session)
                    st.download_button(
                        "📥 Scarica JSON Completo",
                        json_data,
                        f"session_{last_session['id']}.json",
                        "application/json",
                        use_container_width=True
                    )
        else:
            st.info("ℹ️ Nessuna sessione registrata. Avvia una sessione dal tab Live.")

    # TAB 3: STORICO
    with tab_history:
        if st.session_state.sessions_archive:
            st.subheader(f"📚 Storico Sessioni ({len(st.session_state.sessions_archive)} totali)")
            
            summary_data = []
            for s in st.session_state.sessions_archive:
                summary_data.append({
                    'ID': s['id'],
                    'Data': s['date'],
                    'Durata (s)': round(s['summary']['duration'], 1),
                    'Stress Medio': round(s['summary']['stress_avg'], 1),
                    '% Parlato': round(s['summary']['pct_talking'], 1),
                    'Fixations': s['summary']['fixations']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            st.divider()
            
            for s in reversed(st.session_state.sessions_archive):
                with st.expander(
                    f"🔍 Sessione #{s['id']} - {s['date']} (Durata: {s['summary']['duration']:.1f}s)"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Stress", f"{s['summary']['stress_avg']:.1f}/100")
                    c2.metric("Parlato", f"{s['summary']['pct_talking']:.1f}%")
                    c3.metric("Fixations", s['summary']['fixations'])
                    
                    st.dataframe(s['full_data'].head(100), use_container_width=True)
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        csv = s['full_data'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 CSV", 
                            csv, 
                            f"session_{s['id']}.csv",
                            key=f"csv_{s['id']}"
                        )
                    with col_d2:
                        json_data = export_session_json(s)
                        st.download_button(
                            "📥 JSON",
                            json_data,
                            f"session_{s['id']}.json",
                            key=f"json_{s['id']}"
                        )
        else:
            st.info("ℹ️ Nessuna sessione nello storico.")

if __name__ == "__main__":
    main()