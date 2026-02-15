import cv2
import numpy as np
import mediapipe as mp
import json
import time
from confluent_kafka import Producer

class EyeTracker:
    # --- VISUAL CONFIGURATION ---
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    L_EYE_EAR = [362, 385, 387, 263, 373, 380]
    R_EYE_EAR = [33, 160, 158, 133, 153, 144]
    
    L_INNER, L_OUTER = [133], [33]
    R_INNER, R_OUTER = [362], [263]
    L_IRIS, R_IRIS = [468], [473]
    L_TOP, L_BOT = [159], [145]
    R_TOP, R_BOT = [386], [374]

    def __init__(self, blink_threshold=0.21, mouth_threshold=0.2):
        print("Initializing EyeTracker...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.blink_threshold = blink_threshold
        self.mouth_threshold = mouth_threshold
        self.eyes_closed = False
        self.blink_total = 0
        self.kafka_topic = 'gaze_data'
        
        self.x_sensitivity, self.y_sensitivity = 3.8, 4.2
        self.history_x, self.history_y = [], []
        self.smooth_size = 3

        self.producer = None
        try:
            conf = {'bootstrap.servers': 'localhost:9092'}
            self.producer = Producer(conf)
            print("TRACKER: Connected to Kafka successfully!")
        except Exception as e:
            print(f"TRACKER: Kafka Error: {e}")

    def reset_counters(self):
        self.blink_total = 0
        self.eyes_closed = False
        self.history_x, self.history_y = [], []

    def _get_pt(self, landmarks, idx, w, h):
        return np.array([landmarks[idx[0]].x * w, landmarks[idx[0]].y * h])

    def _calculate_ear(self, landmarks, indices, w, h):
        p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        horiz = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Base Data 
        data = {'face_detected': False, 'timestamp': time.time(), 'gaze_x': 0.5, 'gaze_y': 0.5,
                'gaze_horizontal': 'center', 'gaze_vertical': 'center', 'is_talking': False, 
                'blink_detected': False, 'total_blinks': self.blink_total, 'avg_ear': 0.0}

        if not results.multi_face_landmarks:
            return frame, data

        data['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark

        # --- CALCULATIONS (Blink, Gaze, Talking) ---
        ear_l = self._calculate_ear(landmarks, self.L_EYE_EAR, w, h)
        ear_r = self._calculate_ear(landmarks, self.R_EYE_EAR, w, h)
        avg_ear = (ear_l + ear_r) / 2.0
        data['avg_ear'] = round(avg_ear, 4)

        if avg_ear < self.blink_threshold:
            if not self.eyes_closed: self.eyes_closed = True
        else:
            if self.eyes_closed:
                self.eyes_closed = False
                self.blink_total += 1
                data['blink_detected'] = True
        data['total_blinks'] = self.blink_total

        m_top = self._get_pt(landmarks, [13], w, h)
        m_bot = self._get_pt(landmarks, [14], w, h)
        data['is_talking'] = bool(np.linalg.norm(m_top - m_bot) / h > 0.012)

        l_iris = self._get_pt(landmarks, self.L_IRIS, w, h)
        l_in, l_out = self._get_pt(landmarks, self.L_INNER, w, h), self._get_pt(landmarks, self.L_OUTER, w, h)
        l_top, l_bot = self._get_pt(landmarks, self.L_TOP, w, h), self._get_pt(landmarks, self.L_BOT, w, h)
        
        raw_x = (np.linalg.norm(l_iris - l_in) / np.linalg.norm(l_in - l_out)) if np.linalg.norm(l_in - l_out) > 0 else 0.5
        raw_y = (np.linalg.norm(l_iris - l_top) / np.linalg.norm(l_top - l_bot)) if np.linalg.norm(l_top - l_bot) > 0 else 0.5
        
        adj_x = np.clip(0.5 + (raw_x - 0.5) * self.x_sensitivity, 0, 1)
        adj_y = np.clip(0.5 + (raw_y - 0.5) * self.y_sensitivity, 0, 1)
        
        self.history_x.append(adj_x); self.history_y.append(adj_y)
        if len(self.history_x) > self.smooth_size: self.history_x.pop(0); self.history_y.pop(0)
        fx, fy = np.mean(self.history_x), np.mean(self.history_y)
        data['gaze_x'], data['gaze_y'] = round(float(fx), 3), round(float(fy), 3)

        if fx < 0.44: data['gaze_horizontal'] = 'right' 
        elif fx > 0.56: data['gaze_horizontal'] = 'left'
        else: data['gaze_horizontal'] = 'center'

        if fy < 0.42: data['gaze_vertical'] = 'up'
        elif fy > 0.58: data['gaze_vertical'] = 'down'
        else: data['gaze_vertical'] = 'center'

        # --- MESH DRAWING ---
        for contour in [self.LEFT_EYE_CONTOUR, self.RIGHT_EYE_CONTOUR]:
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in contour])
            cv2.polylines(frame, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
        l_px, r_px = tuple(l_iris.astype(int)), tuple(self._get_pt(landmarks, self.R_IRIS, w, h).astype(int))
        for idx in [self.L_INNER, self.L_OUTER, self.L_TOP, self.L_BOT]:
            cv2.line(frame, l_px, tuple(self._get_pt(landmarks, idx, w, h).astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
        for idx in [self.R_INNER, self.R_OUTER, self.R_TOP, self.R_BOT]:
            cv2.line(frame, r_px, tuple(self._get_pt(landmarks, idx, w, h).astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, l_px, 2, (0, 255, 255), -1)
        cv2.circle(frame, r_px, 2, (0, 255, 255), -1)

        # --- KAFKA SEND ---
        if self.producer:
            try:
                kafka_payload = {
                    "timestamp": data['timestamp'],
                    "type": "REALTIME_DATA",
                    "metrics": data
                }
                self.producer.produce(self.kafka_topic, json.dumps(kafka_payload).encode('utf-8'))
                self.producer.poll(0)
            except Exception as e:
                pass

        return frame, data