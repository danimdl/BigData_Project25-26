import cv2
import numpy as np
import mediapipe as mp
import json
import time
from confluent_kafka import Producer

class EyeTracker:
    """
    Classe per il tracking di occhi, sguardo, blink e parlato.
    Restituisce solo dati RAW per frame - NO calcolo metriche aggregate.
    """
    
    # Contorni occhi completi
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Punti per il tracciamento
    L_INNER, L_OUTER = [133], [33]
    R_INNER, R_OUTER = [362], [263]
    L_IRIS, R_IRIS = [468], [473]
    L_TOP, L_BOT = [159], [145]

    # Punti per EAR (Eye Aspect Ratio)
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

    def __init__(self, blink_threshold=0.21, mouth_threshold=0.012, kafka_enabled=False):
        """
        Inizializza l'eye tracker.
        
        Args:
            blink_threshold: Soglia EAR per rilevare i blink
            mouth_threshold: Soglia per rilevare il parlato
            kafka_enabled: Se True, invia dati a Kafka
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        
        self.blink_threshold = blink_threshold
        self.mouth_threshold = mouth_threshold
        self.eyes_closed = False
        self.blink_total = 0
        
        # Sensibilità gaze
        self.x_sensitivity = 3.8
        self.y_sensitivity = 4.2
        self.history_x, self.history_y = [], []
        self.smooth_size = 3
        
        # Per visualizzazione stress (aggiornato dall'esterno)
        self.display_stress_score = 0
        self.display_stress_level = "basso"
        
        # Kafka
        self.kafka_enabled = kafka_enabled
        self.kafka_topic = 'gaze_data'
        if kafka_enabled:
            try:
                self.producer = Producer({'bootstrap.servers': 'localhost:9092'})
            except Exception as e:
                print(f"Kafka init error: {e}")
                self.producer = None
        else:
            self.producer = None

    def _get_pt(self, landmarks, idx, w, h):
        """Ottieni un singolo punto dalle landmarks"""
        return np.array([landmarks[idx[0]].x * w, landmarks[idx[0]].y * h])

    def _get_multi_pts(self, landmarks, indices, w, h):
        """Ottieni multipli punti dalle landmarks"""
        return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])

    def _calculate_ear(self, landmarks, eye_indices, w, h):
        """Calcola Eye Aspect Ratio per rilevare i blink"""
        p1 = self._get_pt(landmarks, [eye_indices[0]], w, h)
        p2 = self._get_pt(landmarks, [eye_indices[1]], w, h)
        p3 = self._get_pt(landmarks, [eye_indices[2]], w, h)
        p4 = self._get_pt(landmarks, [eye_indices[3]], w, h)
        p5 = self._get_pt(landmarks, [eye_indices[4]], w, h)
        p6 = self._get_pt(landmarks, [eye_indices[5]], w, h)
        
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal > 0:
            ear = (vertical1 + vertical2) / (2.0 * horizontal)
        else:
            ear = 0.0
            
        return ear

    def update_stress_display(self, stress_score, stress_level):
        """
        Aggiorna i valori di stress da mostrare sul frame.
        Chiamato dall'applicazione principale dopo il calcolo delle metriche.
        
        Args:
            stress_score: Score numerico 0-100
            stress_level: Livello descrittivo (basso, moderato, alto, etc.)
        """
        self.display_stress_score = stress_score
        self.display_stress_level = stress_level

    def reset_counters(self):
        """Reset di tutti i contatori"""
        self.blink_total = 0
        self.eyes_closed = False
        self.history_x, self.history_y = [], []
        self.display_stress_score = 0
        self.display_stress_level = "basso"

    def process_frame(self, frame):
        """
        Processa un singolo frame e restituisce frame annotato + dati raw.
        
        Args:
            frame: Frame BGR da OpenCV
            
        Returns:
            tuple: (frame_annotato, dati_raw_dict)
        """
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Dati di default
        data = {
            'face_detected': False, 
            'timestamp': time.time(), 
            'gaze_x': 0.5, 
            'gaze_y': 0.5,
            'gaze_horizontal': 'centro', 
            'gaze_vertical': 'centro', 
            'is_talking': False, 
            'blink_detected': False, 
            'total_blinks': self.blink_total, 
            'avg_ear': 0.0
        }

        if not results.multi_face_landmarks:
            return frame, data

        data['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark

        # ===== CALCOLO GAZE =====
        l_iris = self._get_pt(landmarks, self.L_IRIS, w, h)
        l_in = self._get_pt(landmarks, self.L_INNER, w, h)
        l_out = self._get_pt(landmarks, self.L_OUTER, w, h)
        l_top = self._get_pt(landmarks, self.L_TOP, w, h)
        l_bot = self._get_pt(landmarks, self.L_BOT, w, h)

        h_dist = np.linalg.norm(l_in - l_out)
        v_dist = np.linalg.norm(l_top - l_bot)
        
        raw_x = (np.linalg.norm(l_iris - l_in) / h_dist) if h_dist > 0 else 0.5
        raw_y = (np.linalg.norm(l_iris - l_top) / v_dist) if v_dist > 0 else 0.5
        
        adj_x = 0.5 + (raw_x - 0.5) * self.x_sensitivity
        adj_y = 0.5 + (raw_y - 0.5) * self.y_sensitivity
        
        self.history_x.append(adj_x)
        self.history_y.append(adj_y)
        if len(self.history_x) > self.smooth_size:
            self.history_x.pop(0)
            self.history_y.pop(0)
        
        fx = np.clip(np.mean(self.history_x), 0, 1)
        fy = np.clip(np.mean(self.history_y), 0, 1)
        
        data['gaze_x'] = round(float(fx), 3)
        data['gaze_y'] = round(float(fy), 3)

        # Direzione gaze
        if fx < 0.44:
            data['gaze_horizontal'] = 'destra'
        elif fx > 0.56:
            data['gaze_horizontal'] = 'sinistra'
        
        if fy < 0.42:
            data['gaze_vertical'] = 'su'
        elif fy > 0.58:
            data['gaze_vertical'] = 'giu'

        # ===== CALCOLO BLINK =====
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_EAR, w, h)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        data['avg_ear'] = round(float(avg_ear), 3)

        # Rilevamento blink
        if avg_ear < self.blink_threshold:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.blink_total += 1
                data['blink_detected'] = True
        else:
            self.eyes_closed = False
        
        data['total_blinks'] = self.blink_total

        # ===== RILEVAMENTO PARLATO =====
        m_top = self._get_pt(landmarks, [13], w, h)
        m_bot = self._get_pt(landmarks, [14], w, h)
        data['is_talking'] = bool(np.linalg.norm(m_top - m_bot) / h > 0.012)

        # ===== DISEGNO MESH =====
        left_eye_pts = self._get_multi_pts(landmarks, self.LEFT_EYE_CONTOUR, w, h)
        right_eye_pts = self._get_multi_pts(landmarks, self.RIGHT_EYE_CONTOUR, w, h)
        
        cv2.polylines(frame, [left_eye_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [right_eye_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        
        r_iris = self._get_pt(landmarks, self.R_IRIS, w, h)
        cv2.circle(frame, tuple(l_iris.astype(int)), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(r_iris.astype(int)), 3, (0, 255, 255), -1)
        
        cv2.circle(frame, tuple(l_in.astype(int)), 2, (0, 0, 255), -1)
        cv2.circle(frame, tuple(l_out.astype(int)), 2, (0, 0, 255), -1)
        
        # ===== INFO SUL FRAME =====
        cv2.putText(frame, f"Blinks: {self.blink_total}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: ({fx:.2f}, {fy:.2f})", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Stress (usa valori aggiornati dall'esterno)
        stress_color = (
            (0, 255, 0) if self.display_stress_score < 40 
            else (0, 165, 255) if self.display_stress_score < 70 
            else (0, 0, 255)
        )
        cv2.putText(
            frame, 
            f"Stress: {self.display_stress_score:.0f}/100 ({self.display_stress_level})", 
            (10, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            stress_color, 
            2
        )

        # ===== KAFKA (opzionale) =====
        if self.producer:
            try:
                self.producer.produce(self.kafka_topic, json.dumps(data).encode('utf-8'))
                self.producer.poll(0)
            except Exception as e:
                print(f"Errore Kafka: {e}")

        return frame, data