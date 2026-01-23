import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime
import time

class EyeTracker:
    # Indici originali corretti
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

    def __init__(self, smoothing_window=5, blink_threshold=0.21):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
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
        if eye_width == 0 or eye_height == 0: return 0.5, 0.5
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
            'face_detected': False,
            'gaze_x': 0.5, 'gaze_y': 0.5,
            'gaze_horizontal': 'centro', 'gaze_vertical': 'centro',
            'avg_ear': 0.0, 'blink_detected': False, 'total_blinks': self.blink_total,
            'saccade_distance': 0.0,
        }

        if not results.multi_face_landmarks:
            return frame, data

        data['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Iris & Smoothing
        l_iris = self._smooth_coordinates(self._get_iris_center(landmarks, self.LEFT_IRIS, frame.shape), self.left_iris_history)
        r_iris = self._smooth_coordinates(self._get_iris_center(landmarks, self.RIGHT_IRIS, frame.shape), self.right_iris_history)
        
        # EAR & Blink (Logica originale: conta solo quando riapre)
        avg_ear = (self._calculate_ear(landmarks, self.LEFT_EYE_EAR, frame.shape) + 
                   self._calculate_ear(landmarks, self.RIGHT_EYE_EAR, frame.shape)) / 2
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
        l_reg = self._get_eye_region(landmarks, self.LEFT_EYE, frame.shape)
        r_reg = self._get_eye_region(landmarks, self.RIGHT_EYE, frame.shape)
        l_rat = self._calculate_gaze_ratio(l_iris, l_reg)
        r_rat = self._calculate_gaze_ratio(r_iris, r_reg)
        
        ax, ay = (l_rat[0] + r_rat[0]) / 2, (l_rat[1] + r_rat[1]) / 2
        data['gaze_x'], data['gaze_y'] = round(ax, 2), round(ay, 2)
        data['gaze_horizontal'], data['gaze_vertical'] = self._determine_gaze_direction(ax, ay)
        
        # Saccade
        data['saccade_distance'] = round(np.sqrt((ax - self.last_gaze[0])**2 + (ay - self.last_gaze[1])**2), 4)
        self.last_gaze = (ax, ay)

        # Disegno Occhi (Verde + Fucsia)
        l_pts = self._get_landmark_coords(landmarks, self.LEFT_EYE, frame.shape)
        r_pts = self._get_landmark_coords(landmarks, self.RIGHT_EYE, frame.shape)
        cv2.polylines(frame, [l_pts], True, (0, 255, 0), 1)
        cv2.polylines(frame, [r_pts], True, (0, 255, 0), 1)
        cv2.circle(frame, tuple(l_iris), 4, (255, 0, 255), -1)
        cv2.circle(frame, tuple(r_iris), 4, (255, 0, 255), -1)

        return frame, data