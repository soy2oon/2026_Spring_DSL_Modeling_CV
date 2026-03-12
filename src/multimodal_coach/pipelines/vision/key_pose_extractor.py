import numpy as np
import math
from collections import deque

class KeyPoseExtractor:
    """
    Identifies and evaluates critical moments in speech by analyzing the velocity
    of wrist landmarks. Extracts keyframes (peaks or holds) and compares them
    against a reference video using Cosine Similarity.
    """
    def __init__(self, fps=30, velocity_window=5, peak_threshold=0.02, hold_threshold=0.005):
        """
        Args:
            fps (int): Frames per second of the video/webcam feed.
            velocity_window (int): Number of frames over which to smooth velocity calculations.
            peak_threshold (float): Minimum velocity required to trigger a "strong gesture" keyframe.
            hold_threshold (float): Maximum velocity after a peak to trigger a "hold pose" keyframe.
        """
        self.fps = fps
        self.velocity_window = velocity_window
        self.peak_threshold = peak_threshold
        self.hold_threshold = hold_threshold
        
        # State tracking
        self.frame_count = 0
        self.recent_points = {"left": deque(maxlen=velocity_window), "right": deque(maxlen=velocity_window)}
        self.recent_velocities = {"left": 0.0, "right": 0.0}
        
        # Keyframe logging state
        self.in_peak_phase = {"left": False, "right": False}
        self.extracted_keyframes = []

    def _calculate_velocity(self, points_queue) -> float:
        """Calculates smoothed velocity over the recent points queue."""
        if len(points_queue) < self.velocity_window:
            return 0.0
            
        # Distance from oldest to newest point in the window
        p1 = points_queue[0]
        p2 = points_queue[-1]
        
        squared_dist = sum((a - b)**2 for a, b in zip(p1, p2))
        dist = math.sqrt(squared_dist)
            
        time_elapsed = self.velocity_window / self.fps
        return dist / time_elapsed if time_elapsed > 0 else 0.0

    def _cosine_similarity(self, vector_a, vector_b) -> float:
        """Calculates cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))

    def _vectorize_shoulder_to_wrist(self, shoulder, wrist) -> np.ndarray:
        """Creates a vector from shoulder to wrist."""
        return np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1], wrist[2] - shoulder[2]])

    def process_frame(self, user_landmarks, timestamp_ms, ref_frame_data=None):
        """
        Processes a single frame, updates velocity tracking, and extracts keyframes.
        
        Args:
            user_landmarks (dict): Dictionary mapping landmark names (e.g., 'left_wrist') to (x, y, z) tuples.
            timestamp_ms (float): Current playback/recording time in milliseconds.
            ref_frame_data (dict): Reference data dictionary containing synced reference landmarks.
            
        Returns:
            list: A list of string feedback logs generated in this frame. Empty if no keyframe triggered.
        """
        feedback_logs = []
        self.frame_count += 1
        
        for side in ["left", "right"]:
            wrist_key = f"{side}_wrist"
            shoulder_key = f"{side}_shoulder"
            
            if wrist_key not in user_landmarks or shoulder_key not in user_landmarks:
                continue
                
            wrist_pos = user_landmarks[wrist_key]
            shoulder_pos = user_landmarks[shoulder_key]
            
            self.recent_points[side].append(wrist_pos)
            velocity = self._calculate_velocity(self.recent_points[side])
            self.recent_velocities[side] = velocity
            
            # Formatted time string (MM:SS)
            seconds = int(timestamp_ms / 1000)
            time_str = f"{seconds//60:02d}:{seconds%60:02d}"
            
            # --- Keyframe Evaluation Logic ---
            
            # 1. Detect Peak (Strong Gesture Phase)
            if velocity > self.peak_threshold:
                if not self.in_peak_phase[side]:
                    self.in_peak_phase[side] = True # Entered a high-velocity movement phase
                    
            # 2. Detect Drop (Holding a Pose after a movement)
            elif velocity < self.hold_threshold and self.in_peak_phase[side]:
                # The gesture has stopped, this is a keyframe!
                self.in_peak_phase[side] = False
                
                # If we have reference data synced to this timestamp, evaluate it
                if ref_frame_data:
                    user_vector = self._vectorize_shoulder_to_wrist(shoulder_pos, wrist_pos)
                    
                    # Assuming ref_frame_data contains pre-calculated 'shoulder_elbow_wrist_vectors'
                    # specifically [shoulder_to_elbow, elbow_to_wrist]. We can approximate full vector by summing them.
                    ref_vectors = ref_frame_data.get("shoulder_elbow_wrist_vectors", {}).get(side, [])
                    if len(ref_vectors) == 2:
                        ref_vector = np.array(ref_vectors[0]) + np.array(ref_vectors[1])
                    else:
                        ref_vector = np.array([0, -1, 0]) # Fallback dummy vector
                        
                        sim = self._cosine_similarity(user_vector, ref_vector)
                        
                        self.extracted_keyframes.append({
                            "time": time_str,
                            "side": side,
                            "type": "pose_hold",
                            "similarity": sim
                        })
                        
                        # Generate feedback based on similarity
                        if sim < 0.7:
                            # Heuristic analysis (e.g., assessing "narrowness" by looking at horizontal x component vs vertical y)
                            # This is a simplified example; deeper 3D analysis is needed for perfect "narrow" vs "wide"
                            user_width = abs(user_vector[0])
                            ref_width = abs(ref_vector[0])
                            
                            if user_width < ref_width * 0.8:
                                feedback_logs.append(f"At {time_str}, your {side} hand gesture was too narrow compared to the reference. (Sim: {sim*100:.0f}%)")
                            else:
                                feedback_logs.append(f"At {time_str}, your {side} hand gesture angle didn't match the reference. (Sim: {sim*100:.0f}%)")
                        else:
                            feedback_logs.append(f"At {time_str}, excellent {side} gesture hold! (Sim: {sim*100:.0f}%)")

        return feedback_logs

# ==============================================================================
# Example Usage / Unit Test Logic
# ==============================================================================
if __name__ == "__main__":
    extractor = KeyPoseExtractor(fps=30, velocity_window=3)
    
    print("Testing KeyPoseExtractor Pipeline...")
    
    # Dummy Reference Data (Expected format from Test3 Json)
    dummy_ref = {
        "shoulder_elbow_wrist_vectors": {
            "right": [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]] # Diagonal down-right
        }
    }
    
    # Fill buffer with idle
    for i in range(3):
        extractor.process_frame({"right_wrist": (0,0,0), "right_shoulder": (0,1,0)}, 1000 + i*33, dummy_ref)
    
    print("Frame 3 (Idle): Velocity=", extractor.recent_velocities["right"])
    
    # Fast movement (Triggers Peak Phase) -> simulating 0.1 normalized units per frame
    extractor.process_frame({"right_wrist": (0.1,0,0), "right_shoulder": (0,1,0)}, 1100, dummy_ref)
    extractor.process_frame({"right_wrist": (0.2,0,0), "right_shoulder": (0,1,0)}, 1133, dummy_ref)
    extractor.process_frame({"right_wrist": (0.3,0,0), "right_shoulder": (0,1,0)}, 1166, dummy_ref)
    
    print("Frame 6 (Moving Fast): Velocity=", extractor.recent_velocities["right"])
    print(f"  Right side in peak phase: {extractor.in_peak_phase['right']}")
    
    # Sudden Stop (Triggers Hold Keyframe extraction)
    # The wrist stays at (0.3,0,0) for several frames, dropping velocity
    extractor.process_frame({"right_wrist": (0.3,0,0), "right_shoulder": (0,1,0)}, 1200, dummy_ref)
    print("Frame 7 (Slowing): Velocity=", extractor.recent_velocities["right"])
    
    extractor.process_frame({"right_wrist": (0.3,0,0), "right_shoulder": (0,1,0)}, 1233, dummy_ref)
    print("Frame 8 (Slowing): Velocity=", extractor.recent_velocities["right"])
    
    res = extractor.process_frame({"right_wrist": (0.3,0,0), "right_shoulder": (0,1,0)}, 1266, dummy_ref)
    print("Frame 9 (Holding Pose): Velocity=", extractor.recent_velocities["right"])
    print("Logs Generated:", res)
    
    print("\nSummary of Extracted Keyframes:")
    for kf in extractor.extracted_keyframes:
        print(kf)
