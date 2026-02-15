import pandas as pd
import numpy as np

# ========== SUPPORT FUNCTIONS ==========

def add_relative_time(data_list):
    """
    Adds a 'relative_time' field to each element in the list.
    Necessary to calculate session duration from the start.
    """
    if not data_list:
        return data_list
    
    start_time = data_list[0]['timestamp']
    for item in data_list:
        item['relative_time'] = item['timestamp'] - start_time
    
    return data_list


def calculate_saccade_distance(data_list):
    """
    Calculates saccadic distance (gaze movement) between consecutive frames.
    """
    if len(data_list) < 2:
        if data_list:
            data_list[0]['saccade_distance'] = 0.0
        return data_list
    
    data_list[0]['saccade_distance'] = 0.0
    
    for i in range(1, len(data_list)):
        prev_x = data_list[i-1].get('gaze_x', 0.5)
        prev_y = data_list[i-1].get('gaze_y', 0.5)
        curr_x = data_list[i].get('gaze_x', 0.5)
        curr_y = data_list[i].get('gaze_y', 0.5)
        
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        data_list[i]['saccade_distance'] = round(distance, 4)
    
    return data_list


# ========== REALISTIC STRESS CALCULATION ==========

def calculate_stress_score(bpm, variability, velocity, pct_down):
    """
    Calculates a REALISTIC stress score based on eye-tracking parameters.
    
    NORMAL VALUES:
    - Blink rate: 12-20/min (resting normal)
    - Variability: 0.1-0.25 (normal)
    - Velocity: 0.1-0.4 (normal)
    - Pct_down: 0-30% (normal)
    
    LOGIC:
    - Small deviations = Low stress
    - Medium deviations = Moderate stress
    - Large deviations = High stress
    
    Args:
        bpm: Blinks per minute
        variability: Gaze variability (0-1)
        velocity: Scan path velocity
        pct_down: Percentage of looking down
        
    Returns:
        int: Stress score 0-100
    """
    score = 0
    
    # === BLINK RATE (0-30 points) ===
    # Normal: 12-20/min
    if bpm > 35:  # Extremely high
        score += 30
    elif bpm > 28:  # Very high
        score += 20
    elif bpm > 23:  # High
        score += 10
    elif bpm < 5 and bpm > 0:  # Too low (extreme concentration/freeze)
        score += 15
    elif bpm < 8:
        score += 8
    # Range 8-23 = normal
    
    # === GAZE VARIABILITY (0-30 points) ===
    # Normal: 0.1-0.25
    if variability > 0.5:  # Extremely unstable
        score += 30
    elif variability > 0.4:  # Very unstable
        score += 20
    elif variability > 0.3:  # Unstable
        score += 10
    elif variability < 0.05:  # Too fixed (staring/freeze)
        score += 5
    # Range 0.05-0.3 = normal
    
    # === SCAN VELOCITY (0-25 points) ===
    # Normal: 0.1-0.4
    if velocity > 0.8:  # Extremely fast
        score += 25
    elif velocity > 0.6:  # Very fast
        score += 15
    elif velocity > 0.5:  # Fast
        score += 8
    # Range 0-0.5 = normal
    
    # === LOOKING DOWN (0-15 points) ===
    # Normal: 0-30%
    if pct_down > 60:  # Extremely high (avoidance)
        score += 15
    elif pct_down > 50:  # Very high
        score += 10
    elif pct_down > 40:  # High
        score += 5
    # Range 0-40% = normal
    
    return min(100, max(0, score))


def get_stress_level(stress_score):
    """
    Converts numeric stress score to descriptive level.
    """
    if stress_score < 15:
        return "Low"
    elif stress_score < 30:
        return "Moderate-Low"
    elif stress_score < 50:
        return "Moderate"
    elif stress_score < 70:
        return "Moderate-High"
    else:
        return "High"


# ========== METRICS ==========

def get_empty_metrics():
    """Returns a dictionary of empty/default metrics"""
    return {
        'duration_seconds': 0,
        'total_frames': 0,
        'total_blinks': 0,
        'blinks_per_minute': 0,
        'scan_path_length': 0,
        'scan_path_velocity': 0,
        'gaze_x_mean': 0.5,
        'gaze_y_mean': 0.5,
        'gaze_variability': 0,
        'fixation_count': 0,
        'avg_ear': 0,
        'pct_looking_left': 0,
        'pct_looking_center': 0,
        'pct_looking_right': 0,
        'pct_looking_up': 0,
        'pct_looking_down': 0,
        'pct_talking': 0,
        'stress_score': 0,
        'stress_level': 'Low'
    }


def calculate_metrics(data_list):
    """
    Calculates all aggregated metrics from a list of frames.
    
    Args:
        data_list: List of dictionaries containing frame data
        
    Returns:
        dict: Complete dictionary of calculated metrics
    """
    if not data_list:
        return get_empty_metrics()
    
    df = pd.DataFrame(data_list)
    
    df = df[df['face_detected'] == True]
    
    if len(df) == 0:
        return get_empty_metrics()
    
    # === DURATION ===
    duration = df['relative_time'].max() if 'relative_time' in df.columns else 0.1
    if duration <= 0:
        duration = 0.1
    
    # === BLINK ===
    total_blinks = int(df['blink_detected'].sum()) if 'blink_detected' in df.columns else 0
    bpm = (total_blinks / duration * 60) if duration > 0 else 0
    
    # === GAZE VARIABILITY ===
    gx_std = df['gaze_x'].std() if 'gaze_x' in df.columns else 0
    gy_std = df['gaze_y'].std() if 'gaze_y' in df.columns else 0
    variability = np.sqrt(gx_std**2 + gy_std**2) if not np.isnan(gx_std) and not np.isnan(gy_std) else 0
    
    # === SCAN PATH ===
    scan_path = df['saccade_distance'].sum() if 'saccade_distance' in df.columns else 0
    scan_velocity = scan_path / duration if duration > 0 else 0
    
    h_counts = df['gaze_horizontal'].value_counts(normalize=True) * 100 if 'gaze_horizontal' in df.columns else pd.Series()
    v_counts = df['gaze_vertical'].value_counts(normalize=True) * 100 if 'gaze_vertical' in df.columns else pd.Series()
    
    # === TALKING ===
    talking_counts = df['is_talking'].value_counts(normalize=True) * 100 if 'is_talking' in df.columns else pd.Series()
    pct_talking = talking_counts.get(True, 0)
    
    # === FIXATION COUNT ===
    fixation_count = 0
    consecutive = 0
    if 'saccade_distance' in df.columns:
        for s in df['saccade_distance']:
            if s < 0.02:
                consecutive += 1
            else:
                if consecutive >= 3:
                    fixation_count += 1
                consecutive = 0
        if consecutive >= 3:
            fixation_count += 1
    
    # === AVG EAR ===
    avg_ear = df['avg_ear'].mean() if 'avg_ear' in df.columns else 0
    
    # === STRESS CALCULATION ===
    pct_down = v_counts.get('down', 0)
    stress_score = calculate_stress_score(bpm, variability, scan_velocity, pct_down)
    stress_level = get_stress_level(stress_score)
    
    # === RETURN FULL METRICS ===
    return {
        'duration_seconds': round(duration, 2),
        'total_frames': len(df),
        'total_blinks': total_blinks,
        'blinks_per_minute': round(bpm, 2),
        'scan_path_length': round(scan_path, 4),
        'scan_path_velocity': round(scan_velocity, 4),
        'gaze_x_mean': round(df['gaze_x'].mean(), 4) if 'gaze_x' in df.columns else 0.5,
        'gaze_y_mean': round(df['gaze_y'].mean(), 4) if 'gaze_y' in df.columns else 0.5,
        'gaze_variability': round(variability, 4),
        'fixation_count': fixation_count,
        'avg_ear': round(avg_ear, 4),
        'pct_looking_left': round(h_counts.get('left', 0), 1),
        'pct_looking_center': round(h_counts.get('center', 0), 1),
        'pct_looking_right': round(h_counts.get('right', 0), 1),
        'pct_looking_up': round(v_counts.get('up', 0), 1),
        'pct_looking_down': round(pct_down, 1),
        'pct_talking': round(pct_talking, 1),
        'stress_score': stress_score,
        'stress_level': stress_level,
    }


# ========== UTILITY FUNCTIONS ==========

def process_data_for_metrics(data_list):
    """Processes raw data list adding calculated fields"""
    if not data_list:
        return data_list
    
    data_with_time = add_relative_time(data_list)
    data_with_saccades = calculate_saccade_distance(data_with_time)
    
    return data_with_saccades


def validate_data_point(data_point):
    """Validates that a single data point has all required fields"""
    required_fields = [
        'face_detected', 'timestamp', 'gaze_x', 'gaze_y',
        'blink_detected', 'is_talking', 'avg_ear'
    ]
    
    for field in required_fields:
        if field not in data_point:
            return False
    
    return True


def filter_valid_data(data_list):
    """Filters only valid data points from a list"""
    return [d for d in data_list if validate_data_point(d)]