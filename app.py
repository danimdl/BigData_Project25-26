import streamlit as st
import cv2
import time
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from tracker_engine import EyeTracker
from analytics_utils import calculate_metrics

def main():
    st.set_page_config(page_title="Eye Tracking - Stress Analysis", layout="wide")
    st.title("👁️ Eye Tracking - Analisi Stress")

    if 'data_list' not in st.session_state: st.session_state.data_list = []
    if 'recording' not in st.session_state: st.session_state.recording = False
    if 'session_name' not in st.session_state: 
        st.session_state.session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with st.sidebar:
        st.header("⚙️ Impostazioni")
        blink_th = st.slider("Soglia Blink", 0.15, 0.30, 0.21, 0.01)
        cam_id = st.number_input("Camera ID", 0, 5, 0)
        st.divider()
        st.session_state.session_name = st.text_input("Nome Sessione", st.session_state.session_name)
        if st.button("🗑️ Reset Dati", use_container_width=True):
            st.session_state.data_list = []
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🔴 Registra", "📊 Metriche", "💾 Esporta"])

    with tab1:
        col_video, col_rt = st.columns([2, 1])
        with col_video:
            if not st.session_state.recording:
                if st.button("Avvia Registrazione", type="primary", use_container_width=True):
                    st.session_state.recording = True
                    st.session_state.data_list = []
                    st.rerun()
            else:
                if st.button("STOP Registrazione", type="secondary", use_container_width=True):
                    st.session_state.recording = False
                    st.rerun()
            video_place = st.empty()
            status_bar = st.empty()

        with col_rt:
            st.subheader("📊 Dati Attuali")
            m1, m2, m3, m4, m5, m6 = st.empty(), st.empty(), st.empty(), st.empty(), st.empty(), st.empty()

        if st.session_state.recording:
            cap = cv2.VideoCapture(int(cam_id))
            tracker = EyeTracker(blink_threshold=blink_th)
            start_t = time.time()
            while st.session_state.recording:
                ret, frame = cap.read()
                if not ret: break
                proc_frame, data = tracker.process_frame(frame)
                elapsed = time.time() - start_t
                data['relative_time'] = round(elapsed, 3)
                st.session_state.data_list.append(data)

                video_place.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB))
                status_bar.markdown(f"**REC** | ⏱️ {elapsed:.1f}s | 🖼️ {len(st.session_state.data_list)} frames")
                
                m1.metric("Blink Totali", data['total_blinks'])
                m2.metric("Sguardo", f"{data['gaze_horizontal']}, {data['gaze_vertical']}")
                m3.metric("EAR (Apertura)", f"{data['avg_ear']:.4f}")
                m4.metric("Stress Score", f"{calculate_metrics(st.session_state.data_list)['stress_score']}/100")
                m5.metric("Posizione (X, Y)", f"({data['gaze_x']}, {data['gaze_y']})")
                m6.metric("Saccade", f"{data['saccade_distance']:.4f}")
                time.sleep(0.01)
            cap.release()

    with tab2:
        if st.session_state.data_list:
            metrics = calculate_metrics(st.session_state.data_list)
            df = pd.DataFrame(st.session_state.data_list)
            
            # Card Metriche (Foto 4)
            st.subheader("Analisi Metriche")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Durata", f"{metrics['duration_seconds']}s")
                st.metric("Frame", metrics['total_frames'])
            with c2:
                st.metric("Blink", metrics['total_blinks'])
                st.metric("Blink/min", metrics['blinks_per_minute'])
            with c3:
                st.metric("Scan Path", metrics['scan_path_length'])
                st.metric("Fixations", metrics['fixation_count'])
            with c4:
                st.metric("Stress", f"{metrics['stress_score']}/100")
                st.metric("% Giù", f"{metrics['pct_looking_down']}%")

            st.divider()
            
            # Grafici (Foto 4)
            g1, g2 = st.columns(2)
            with g1:
                st.subheader("Distribuzione Direzioni")
                dir_df = pd.DataFrame({'Direzione': ['Sinistra', 'Centro', 'Destra', 'Su', 'Giù'],
                    'Percentuale': [metrics['pct_looking_left'], metrics['pct_looking_center'], 
                                   metrics['pct_looking_right'], metrics['pct_looking_up'], metrics['pct_looking_down']]})
                st.plotly_chart(px.bar(dir_df, x='Direzione', y='Percentuale', color='Percentuale', color_continuous_scale='RdYlGn_r'), use_container_width=True)
            with g2:
                st.subheader("Heatmap Sguardo")
                fig_hm = px.density_heatmap(df, x='gaze_x', y='gaze_y', nbinsx=15, nbinsy=15, color_continuous_scale='Hot')
                fig_hm.update_layout(xaxis_range=[0,1], yaxis_range=[0,1])
                st.plotly_chart(fig_hm, use_container_width=True)

            st.subheader("Timeline")
            fig_tl = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Posizione Sguardo', 'EAR'))
            fig_tl.add_trace(go.Scatter(x=df['relative_time'], y=df['gaze_x'], name='X', line=dict(color='blue')), row=1, col=1)
            fig_tl.add_trace(go.Scatter(x=df['relative_time'], y=df['gaze_y'], name='Y', line=dict(color='red')), row=1, col=1)
            fig_tl.add_trace(go.Scatter(x=df['relative_time'], y=df['avg_ear'], name='EAR', line=dict(color='green')), row=2, col=1)
            fig_tl.add_hline(y=0.21, line_dash="dash", line_color="red", row=2, col=1)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("Registra dati per visualizzare le analisi.")

    with tab3:
        st.subheader(" Esporta Dati")
        if not st.session_state.data_list:
            st.warning(" Nessun dato da esportare.")
        else:
            df = pd.DataFrame(st.session_state.data_list)
            metrics = calculate_metrics(st.session_state.data_list)
            st.success(f" **{len(df)} frame** pronti per l'export")
            
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button(" Scarica CSV", df.to_csv(index=False), "data.csv", "text/csv", use_container_width=True)
            with c2: st.download_button(" Scarica Metriche JSON", json.dumps(metrics, indent=2), "metrics.json", "application/json", use_container_width=True)
            with c3: st.download_button(" Export Completo", json.dumps({'metrics': metrics, 'data': df.to_dict('records')}, indent=2), "full.json", "application/json", use_container_width=True)
            
            st.divider()
            st.subheader(" Riepilogo Metriche")
            res_df = pd.DataFrame([{'Metrica': k, 'Valore': v} for k, v in metrics.items()])
            st.dataframe(res_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()