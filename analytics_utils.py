import pandas as pd
import numpy as np

def calculate_stress_score(bpm, variability, velocity, pct_down):
    score = 0
    if bpm > 30: score += 25
    elif bpm > 25: score += 15
    if variability > 0.3: score += 25
    if velocity > 0.5: score += 20
    if pct_down > 40: score += 20
    return min(100, score)

def get_empty_metrics():
    return {
        'duration_seconds': 0, 'total_frames': 0, 'total_blinks': 0,
        'blinks_per_minute': 0, 'scan_path_length': 0, 'stress_score': 0,
        'pct_talking': 0, 'avg_ear': 0, 'pct_looking_down': 0
    }

def calculate_metrics(data_list):
    if not data_list: return get_empty_metrics()
    df = pd.DataFrame(data_list)
    df = df[df['face_detected'] == True]
    if len(df) == 0: return get_empty_metrics()
    
    duration = df['relative_time'].max() if 'relative_time' in df else 0.1
    total_blinks = df['blink_detected'].sum()
    bpm = (total_blinks / duration * 60)
    
    gx_std, gy_std = df['gaze_x'].std(), df['gaze_y'].std()
    variability = np.sqrt(gx_std**2 + gy_std**2) if not np.isnan(gx_std) else 0
    scan_path = df['saccade_distance'].sum()
    
    h_counts = df['gaze_horizontal'].value_counts(normalize=True) * 100
    v_counts = df['gaze_vertical'].value_counts(normalize=True) * 100
    
    # Calcolo Parlato
    talking_counts = df['is_talking'].value_counts(normalize=True) * 100
    pct_talking = talking_counts.get(True, 0)
    
    fixation_count, consecutive = 0, 0
    for s in df['saccade_distance']:
        if s < 0.02: consecutive += 1
        else:
            if consecutive >= 3: fixation_count += 1
            consecutive = 0

    return {
        'duration_seconds': round(duration, 2),
        'total_frames': len(df),
        'total_blinks': int(total_blinks),
        'blinks_per_minute': round(bpm, 2),
        'scan_path_length': round(scan_path, 4),
        'scan_path_velocity': round(scan_path / duration, 4),
        'gaze_x_mean': round(df['gaze_x'].mean(), 4),
        'gaze_y_mean': round(df['gaze_y'].mean(), 4),
        'gaze_variability': round(variability, 4),
        'fixation_count': fixation_count,
        'avg_ear': round(df['avg_ear'].mean(), 4),
        'pct_looking_left': round(h_counts.get('sinistra', 0), 1),
        'pct_looking_center': round(h_counts.get('centro', 0), 1),
        'pct_looking_right': round(h_counts.get('destra', 0), 1),
        'pct_looking_up': round(v_counts.get('su', 0), 1),
        'pct_looking_down': round(v_counts.get('giù', 0), 1),
        'pct_talking': round(pct_talking, 1),
        'stress_score': calculate_stress_score(bpm, variability, scan_path/duration, v_counts.get('giù', 0)),
    }