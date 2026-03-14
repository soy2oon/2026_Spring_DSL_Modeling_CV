import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque

class GazeAnxietyDetector:
    """
    Detects interview anxiety through eye movement patterns using MediaPipe Face Mesh.
    Focuses on Gaze Avoidance (looking away from the camera) and Eye Jitter (nervous shaking).
    """
    def __init__(self, 
                 calibration_frames: int = 90, 
                 avoidance_threshold: float = 0.05, 
                 jitter_threshold: float = 0.0001,
                 window_size: int = 15):
        """
        Args:
            calibration_frames: Number of frames to use for initial reference point calibration (~3 secs at 30 fps).
            avoidance_threshold: Distance threshold from reference point to count as avoiding gaze.
            jitter_threshold: Variance threshold in moving average window to count as nervous shaking.
            window_size: Size of the sliding window for calculating variance (jitter).
        """
        # Configurations
        self.calibration_frames = calibration_frames
        self.avoidance_threshold = avoidance_threshold
        self.jitter_threshold = jitter_threshold
        self.window_size = window_size
        
        # State Tracking
        self.is_calibrating = True
        self.calibration_count = 0
        self.reference_point = None  # (mean_x, mean_y) of left and right iris centers combined
        
        # Buffers for temporal analysis
        self.calibration_buffer = [] # Holds points during calibration phase
        self.gaze_history = deque(maxlen=window_size) # Holds recent (x,y) points for jitter detection
        
        # MediaPipe Face Mesh Setup
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True is crucial for detecting the Iris landmarks (468-477)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Iris Landmark Indices
        # Left Iris: 468 (center), 469, 470, 471, 472 (edges)
        # Right Iris: 473 (center), 474, 475, 476, 477 (edges)
        self.LEFT_IRIS_CENTER = 468
        self.RIGHT_IRIS_CENTER = 473

    def _get_iris_center(self, landmarks) -> tuple:
        """
        Extracts the center point of the irises combined.
        Returns a normalized (x, y) tuple representing the mean position of both irises.
        """
        left_iris = landmarks.landmark[self.LEFT_IRIS_CENTER]
        right_iris = landmarks.landmark[self.RIGHT_IRIS_CENTER]
        
        # Take the geometric center of both irises 
        avg_x = (left_iris.x + right_iris.x) / 2.0
        avg_y = (left_iris.y + right_iris.y) / 2.0
        
        return (avg_x, avg_y)
        
    def _calculate_distance(self, p1: tuple, p2: tuple) -> float:
        """Calculates Euclidean distance between two (x,y) normalized points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calibrate(self, current_point: tuple):
        """
        Phase 1: Collects Iris coordinates to establish a baseline REFERENCE_POINT.
        """
        self.calibration_buffer.append(current_point)
        self.calibration_count += 1
        
        if self.calibration_count >= self.calibration_frames:
            # End of calibration phase: Calculate the mean (x,y) over the collected frames
            x_coords = [p[0] for p in self.calibration_buffer]
            y_coords = [p[1] for p in self.calibration_buffer]
            
            self.reference_point = (np.mean(x_coords), np.mean(y_coords))
            self.is_calibrating = False
            self.calibration_buffer.clear() # Free memory
            
    def _evaluate_stability(self, current_point: tuple) -> dict:
        """
        Phase 2 & 3: Evaluates current gaze point against reference and history 
        to determine if the user is avoiding gaze or showing jitter.
        """
        if self.reference_point is None:
            return {"status": "Error: Not Calibrated", "score": 0.0}
            
        # Update history window for temporal analysis
        self.gaze_history.append(current_point)
        
        # Phase 2: Gaze Avoidance Detection (Spatial)
        # Distance from the calibrated "Looking at Camera" point
        spatial_distance = self._calculate_distance(current_point, self.reference_point)
        
        if spatial_distance > self.avoidance_threshold:
            # Score maps distance to a 0.0 - 1.0 confidence score (soft capping at 2x threshold)
            score = min(1.0, spatial_distance / (self.avoidance_threshold * 2))
            return {
                "status": "Avoiding", 
                "message": "Please look at the camera.",
                "score": score,
                "distance": spatial_distance
            }
            
        # Phase 3: Eye Jitter/Tremor Detection (Temporal/Statistical)
        # Only evaluate if we have enough history frames
        if len(self.gaze_history) == self.window_size:
            points_array = np.array(self.gaze_history)
            
            # Apply a simple moving average / low-pass filter over the window
            # Taking the variance of the X and Y coordinates respectively
            var_x = np.var(points_array[:, 0])
            var_y = np.var(points_array[:, 1])
            total_variance = var_x + var_y
            
            if total_variance > self.jitter_threshold:
                # User is generally looking at the camera (passed Phase 2), but eyes are shaking
                score = min(1.0, total_variance / (self.jitter_threshold * 3))
                return {
                    "status": "Shaking",
                    "message": "Eyes are shaking. Take a deep breath.",
                    "score": score,
                    "variance": total_variance
                }
                
        # If no thresholds are breached, user is maintaining stable eye contact
        return {
            "status": "Stable",
            "message": "Good eye contact.",
            "score": 0.0
        }

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        """
        Main entry point. Processes a single BGR image frame (e.g. from cv2.VideoCapture).
        
        Args:
            image_bgr: OpenCV BGR image array.
            
        Returns:
            Dictionary containing the analysis status, string message, and confidence score.
        """
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image_bgr.flags.writeable = False
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        image_bgr.flags.writeable = True
        
        if not results.multi_face_landmarks:
            return {"status": "No Face Detected", "score": 0.0}
            
        # Assume one face per frame
        landmarks = results.multi_face_landmarks[0]
        current_iris_point = self._get_iris_center(landmarks)
        
        # Route logic based on calibration state
        if self.is_calibrating:
            self.calibrate(current_iris_point)
            
            progress_pct = int((self.calibration_count / self.calibration_frames) * 100)
            return {
                "status": "Calibrating",
                "message": f"Looking at Camera for Calibration... {progress_pct}%",
                "score": 0.0
            }
        else:
            return self._evaluate_stability(current_iris_point)
            
    def release(self):
        """Releases the underlying MediaPipe resources."""
        self.face_mesh.close()


# ==============================================================================
# Example Usage Driver
# ==============================================================================
if __name__ == "__main__":
    def run_demo():
        cap = cv2.VideoCapture(0)
        detector = GazeAnxietyDetector(calibration_frames=90) # ~3 seconds at 30 fps
        
        print("Starting Gaze Anxiety Detector Demo...")
        print("Please look straight at the camera to calibrate.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1) # Mirror image for ease of use
            
            # Process Frame
            result = detector.process_frame(frame)
            
            # Rendering UI
            status = result.get('status', 'Unknown')
            msg = result.get('message', '')
            score = result.get('score', 0.0)
            
            # Dynamic Colors based on status
            color = (0, 255, 0) # Green for Stable
            if status == "Calibrating":
                color = (0, 255, 255) # Yellow
            elif status == "Avoiding":
                color = (0, 0, 255) # Red
            elif status == "Shaking":
                color = (0, 165, 255) # Orange (BGR)
                
            cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, msg, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if status in ["Avoiding", "Shaking"]:
                cv2.putText(frame, f"Intensity: {score:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Render visual reference point / boundaries if calibrated
            if not detector.is_calibrating and detector.reference_point:
                h, w = frame.shape[:2]
                rx, ry = int(detector.reference_point[0] * w), int(detector.reference_point[1] * h)
                
                # Draw small circle for reference center
                cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)
                
                # Draw avoidance threshold boundary (approximate representation ignoring aspect ratio stretching)
                threshold_px = int(detector.avoidance_threshold * min(w, h))
                cv2.circle(frame, (rx, ry), threshold_px, (0, 255, 0), 1)

            cv2.imshow("Gaze Anxiety Detector", frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        detector.release()
        cap.release()
        cv2.destroyAllWindows()

    run_demo()
