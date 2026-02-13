import pandas as pd
import numpy as np

# ========== FUNZIONI DI SUPPORTO ==========

def add_relative_time(data_list):
    """
    Aggiunge il campo 'relative_time' a ogni elemento della lista.
    Necessario per calcolare la durata della sessione.
    """
    if not data_list:
        return data_list
    
    start_time = data_list[0]['timestamp']
    for item in data_list:
        item['relative_time'] = item['timestamp'] - start_time
    
    return data_list


def calculate_saccade_distance(data_list):
    """
    Calcola la distanza saccadica (movimento dello sguardo) tra frame consecutivi.
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


# ========== CALCOLO STRESS REALISTICO ==========

def calculate_stress_score(bpm, variability, velocity, pct_down):
    """
    Calcola lo stress score REALISTICO basato su parametri eye-tracking.
    
    VALORI NORMALI:
    - Blink rate: 12-20/min (normale riposo)
    - Variability: 0.1-0.25 (normale)
    - Velocity: 0.1-0.4 (normale)
    - Pct_down: 0-30% (normale)
    
    Logica REALISTICA:
    - Piccole deviazioni dalla norma = stress basso
    - Medie deviazioni = stress moderato
    - Grandi deviazioni = stress alto
    
    Args:
        bpm: Blink per minuto
        variability: Variabilità dello sguardo (0-1)
        velocity: Velocità scan path
        pct_down: Percentuale sguardo verso il basso
        
    Returns:
        int: Stress score 0-100
    """
    score = 0
    
    # === BLINK RATE (0-30 punti) ===
    # Normale: 12-20/min
    if bpm > 35:  # Estremamente alto
        score += 30
    elif bpm > 28:  # Molto alto
        score += 20
    elif bpm > 23:  # Alto
        score += 10
    elif bpm < 5 and bpm > 0:  # Troppo basso (concentrazione estrema)
        score += 15
    elif bpm < 8:
        score += 8
    # Range 8-23 = normale, nessun punteggio
    
    # === VARIABILITÀ SGUARDO (0-30 punti) ===
    # Normale: 0.1-0.25
    if variability > 0.5:  # Estremamente instabile
        score += 30
    elif variability > 0.4:  # Molto instabile
        score += 20
    elif variability > 0.3:  # Instabile
        score += 10
    elif variability < 0.05:  # Troppo fisso (può indicare stress/freeze)
        score += 5
    # Range 0.05-0.3 = normale
    
    # === VELOCITÀ MOVIMENTI (0-25 punti) ===
    # Normale: 0.1-0.4
    if velocity > 0.8:  # Estremamente rapido
        score += 25
    elif velocity > 0.6:  # Molto rapido
        score += 15
    elif velocity > 0.5:  # Rapido
        score += 8
    # Range 0-0.5 = normale
    
    # === SGUARDO VERSO IL BASSO (0-15 punti) ===
    # Normale: 0-30%
    if pct_down > 60:  # Estremamente alto (evitamento)
        score += 15
    elif pct_down > 50:  # Molto alto
        score += 10
    elif pct_down > 40:  # Alto
        score += 5
    # Range 0-40% = normale
    
    return min(100, max(0, score))


def get_stress_level(stress_score):
    """
    Converti stress score numerico in livello descrittivo.
    
    SCALA REALISTICA:
    0-15: basso (rilassato)
    15-30: moderato-basso (leggera tensione)
    30-50: moderato (stress gestibile)
    50-70: moderato-alto (stress significativo)
    70-100: alto (stress elevato)
    """
    if stress_score < 15:
        return "basso"
    elif stress_score < 30:
        return "moderato-basso"
    elif stress_score < 50:
        return "moderato"
    elif stress_score < 70:
        return "moderato-alto"
    else:
        return "alto"


# ========== METRICHE ==========

def get_empty_metrics():
    """Restituisce un dizionario di metriche vuote/default"""
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
        'stress_level': 'basso'
    }


def calculate_metrics(data_list):
    """
    Calcola tutte le metriche aggregate da una lista di frame data.
    
    Args:
        data_list: Lista di dizionari con dati per frame
        
    Returns:
        dict: Dizionario completo di metriche calcolate
    """
    if not data_list:
        return get_empty_metrics()
    
    # Converti in DataFrame
    df = pd.DataFrame(data_list)
    
    # Filtra solo frame con faccia rilevata
    df = df[df['face_detected'] == True]
    
    if len(df) == 0:
        return get_empty_metrics()
    
    # === DURATA ===
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
    
    # === DISTRIBUZIONE GAZE ===
    h_counts = df['gaze_horizontal'].value_counts(normalize=True) * 100 if 'gaze_horizontal' in df.columns else pd.Series()
    v_counts = df['gaze_vertical'].value_counts(normalize=True) * 100 if 'gaze_vertical' in df.columns else pd.Series()
    
    # === PARLATO ===
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
    
    # === EAR MEDIO ===
    avg_ear = df['avg_ear'].mean() if 'avg_ear' in df.columns else 0
    
    # === CALCOLO STRESS REALISTICO ===
    pct_down = v_counts.get('giu', 0)
    stress_score = calculate_stress_score(bpm, variability, scan_velocity, pct_down)
    stress_level = get_stress_level(stress_score)
    
    # === RETURN METRICHE COMPLETE ===
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
        'pct_looking_left': round(h_counts.get('sinistra', 0), 1),
        'pct_looking_center': round(h_counts.get('centro', 0), 1),
        'pct_looking_right': round(h_counts.get('destra', 0), 1),
        'pct_looking_up': round(v_counts.get('su', 0), 1),
        'pct_looking_down': round(pct_down, 1),
        'pct_talking': round(pct_talking, 1),
        'stress_score': stress_score,
        'stress_level': stress_level,
    }


# ========== UTILITY FUNCTIONS ==========

def process_data_for_metrics(data_list):
    """Processa una lista di dati raw aggiungendo campi calcolati necessari"""
    if not data_list:
        return data_list
    
    data_with_time = add_relative_time(data_list)
    data_with_saccades = calculate_saccade_distance(data_with_time)
    
    return data_with_saccades


def validate_data_point(data_point):
    """Valida che un singolo data point abbia tutti i campi necessari"""
    required_fields = [
        'face_detected', 'timestamp', 'gaze_x', 'gaze_y',
        'blink_detected', 'is_talking', 'avg_ear'
    ]
    
    for field in required_fields:
        if field not in data_point:
            return False
    
    return True


def filter_valid_data(data_list):
    """Filtra solo i data point validi da una lista"""
    return [d for d in data_list if validate_data_point(d)]