import cv2
import numpy as np

class SpatialTracker:
    """
    Perception Module: Extracts spatial offsets (e_x, e_y) from video frames
    to construct State Vector S_t for the MDP Decision Engine.
    """
    def __init__(self, frame_width=640, frame_height=360):
        self.width = frame_width
        self.height = frame_height
        self.prev_ex = 0.0
        self.prev_ey = 0.0

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return resized, blurred

    def extract_spatial_telemetry(self, raw_frame):
        processed_frame, blurred = self.preprocess_frame(raw_frame)
        
        # Desired Center Target Coordinates
        center_x, center_y = self.width // 2, self.height // 2

        # Canny Feature Extraction & ROI Masking
        edges = cv2.Canny(blurred, 50, 150)
        roi_mask = np.zeros_like(edges)
        cv2.rectangle(roi_mask, (80, 80), (self.width - 80, self.height - 20), 255, -1)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Centroid Calculation via Image Moments
        M = cv2.moments(masked_edges)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = center_x, center_y

        # Geometrical Error Vectors
        ex = float(cx - center_x)
        ey = float(cy - center_y)

        # Derivatives / Velocity Rates
        ex_dot = ex - self.prev_ex
        ey_dot = ey - self.prev_ey

        self.prev_ex = ex
        self.prev_ey = ey

        # Visual Annotations
        cv2.circle(processed_frame, (center_x, center_y), 6, (0, 0, 255), -1)  # Target Center
        cv2.circle(processed_frame, (cx, cy), 6, (0, 255, 0), -1)              # Detected Object
        cv2.line(processed_frame, (center_x, center_y), (cx, cy), (255, 255, 0), 2)  # Drift Line

        # State Vector S_t
        state_vector = np.array([ex, ey, ex_dot, ey_dot], dtype=np.float32)

        return processed_frame, state_vector