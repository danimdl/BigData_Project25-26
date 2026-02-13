import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

class EyeTracker:
    # Landmark Occhi
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    
    # Landmark Bocca (per rilevamento parlato)
    MOUTH_INNER_TOP_BOTTOM = [13, 14]
    MOUTH_CORNERS = [78, 308]

    def __init__(self, smoothing_window=5, blink_threshold=0.21, mouth_threshold=0.2):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.blink_threshold = blink_threshold
        self.mouth_threshold = mouth_threshold 
        self.eyes_closed = False
        self.blink_total = 0
        self.last_gaze = (0.5, 0.5)

    def _get_landmark_coords(self, landmarks, indices, frame_shape):
        h, w = frame_shape[:2]
        return np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices])

    def _calculate_ear(self, landmarks, eye_indices, frame_shape):
        coords = self._get_landmark_coords(landmarks, eye_indices, frame_shape)
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def _calculate_mar(self, landmarks, frame_shape):
        """Calcola il Mouth Aspect Ratio per rilevare se l'utente parla."""
        lip_coords = self._get_landmark_coords(landmarks, self.MOUTH_INNER_TOP_BOTTOM, frame_shape)
        corner_coords = self._get_landmark_coords(landmarks, self.MOUTH_CORNERS, frame_shape)
        vertical_dist = np.linalg.norm(lip_coords[0] - lip_coords[1])
        horizontal_dist = np.linalg.norm(corner_coords[0] - corner_coords[1])
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        data = {
            'face_detected': False, 'gaze_x': 0.5, 'gaze_y': 0.5,
            'gaze_horizontal': 'centro', 'gaze_vertical': 'centro',
            'avg_ear': 0.0, 'blink_detected': False, 'total_blinks': self.blink_total,
            'saccade_distance': 0.0, 'is_talking': False
        }

        if not results.multi_face_landmarks:
            return frame, data

        data['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. EAR & Blink (Logica: conta solo quando l'occhio riapre)
        avg_ear = (self._calculate_ear(landmarks, self.LEFT_EYE_EAR, frame.shape) + 
                   self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, frame.shape)) / 2
        data['avg_ear'] = round(avg_ear, 4)

        if avg_ear < self.blink_threshold:
            self.eyes_closed = True
        elif self.eyes_closed:
            self.eyes_closed = False
            self.blink_total += 1
            data['blink_detected'] = True
        data['total_blinks'] = self.blink_total

        # 2. MAR & Parlato
        mar = self._calculate_mar(landmarks, frame.shape)
        data['is_talking'] = mar > self.mouth_threshold

        # 3. Sguardo e Saccade
        l_iris = np.mean(self._get_landmark_coords(landmarks, self.LEFT_IRIS, frame.shape), axis=0).astype(int)
        ax, ay = l_iris[0]/frame.shape[1], l_iris[1]/frame.shape[0]
        data['gaze_x'], data['gaze_y'] = round(ax, 2), round(ay, 2)
        data['gaze_horizontal'] = 'destra' if ax < 0.35 else 'sinistra' if ax > 0.65 else 'centro'
        data['gaze_vertical'] = 'su' if ay < 0.35 else 'giù' if ay > 0.65 else 'centro'
        data['saccade_distance'] = round(np.sqrt((ax - self.last_gaze[0])**2 + (ay - self.last_gaze[1])**2), 4)
        self.last_gaze = (ax, ay)

        # Disegno tecnico (Contorno verde + iridi fucsia)
        l_pts = self._get_landmark_coords(landmarks, self.LEFT_EYE, frame.shape)
        r_pts = self._get_landmark_coords(landmarks, self.RIGHT_EYE, frame.shape)
        cv2.polylines(frame, [l_pts], True, (0, 255, 0), 1)
        cv2.polylines(frame, [r_pts], True, (0, 255, 0), 1)
        cv2.circle(frame, tuple(l_iris), 4, (255, 0, 255), -1)
        # Nota: per brevità disegniamo solo il centro dell'iride sinistra come riferimento principale
        r_iris = np.mean(self._get_landmark_coords(landmarks, self.RIGHT_IRIS, frame.shape), axis=0).astype(int)
        cv2.circle(frame, tuple(r_iris), 4, (255, 0, 255), -1)

        return frame, data