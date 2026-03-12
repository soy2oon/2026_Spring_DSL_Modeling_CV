import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PoseComparator:
    """
    A class to compare sequences of skeletal pose landmarks using Dynamic Time Warping (DTW).
    Designed to be robust to speed variations and scale/translation differences.
    """
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size (int): The number of frames considered per comparison segment.
        """
        self.window_size = window_size

    def _preprocess(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Applies Centering and L2 Normalization to the pose data.
        This makes the skeleton invariant to camera distance (scale) and position (translation).
        
        Args:
            pose_data: A numpy array of shape [Frames, Num_Landmarks, 3]
            
        Returns:
            Preprocessed numpy array of the same shape.
        """
        pose_data = np.array(pose_data, dtype=np.float32)
        if pose_data.ndim != 3:
            raise ValueError(f"pose_data must have shape [Frames, Num_Landmarks, 3], got {pose_data.shape}")
            
        normalized = np.zeros_like(pose_data)
        
        for i in range(pose_data.shape[0]):
            frame = pose_data[i]
            
            # 1. Translation Invariance: Center the skeleton around the origin (e.g., its mean position)
            center = np.mean(frame, axis=0)
            centered = frame - center
            
            # 2. Scale Invariance: L2 Normalize the flattened frame
            norm = np.linalg.norm(centered)
            if norm > 1e-6:
                normalized[i] = centered / norm
            else:
                normalized[i] = centered
                
        return normalized

    def compare_realtime(self, user_window: np.ndarray, reference_window: np.ndarray) -> float:
        """
        Compare two segments of pose data (e.g., shape: [30, 33, 3]) using DTW.
        Useful for a real-time sliding window evaluation.
        
        Args:
            user_window: Pose data segment from the user.
            reference_window: Corresponding pose data segment from the reference recording.
            
        Returns:
            float: A similarity score bounded between 0.0 and 1.0 (1.0 = exact match).
        """
        if len(user_window) == 0 or len(reference_window) == 0:
            return 0.0
            
        # Preprocess both sequences to normalize scale and translation
        user_norm = self._preprocess(user_window)
        ref_norm = self._preprocess(reference_window)
        
        # Flatten the spatial dimensions for each frame to compute standard Euclidean distances
        # Shape goes from [Frames, Num_Landmarks, 3] -> [Frames, Num_Landmarks * 3]
        user_flat = user_norm.reshape(user_norm.shape[0], -1)
        ref_flat = ref_norm.reshape(ref_norm.shape[0], -1)
        
        # Calculate DTW using fastdtw (O(N) time complexity)
        # Returns the total cumulative distance and the optimal alignment path
        distance, path = fastdtw(user_flat, ref_flat, dist=euclidean)
        
        # Normalized Path Distance
        # The max Euclidean distance between two L2-normalized vectors is 2.
        # The path length reflects how many pointwise comparisons were made in DTW.
        max_possible_distance = 2.0 * len(path)
        
        if max_possible_distance == 0:
            return 1.0
            
        normalized_distance = distance / max_possible_distance
        
        # Convert path distance to a similarity score (0.0 to 1.0)
        # We clamp to ensure bounds. Non-linear scaling (like exp(-dist)) could also be used here.
        similarity = max(0.0, min(1.0, 1.0 - normalized_distance))
        
        return similarity
        
    def compare_full_sequences(self, user_seq: np.ndarray, reference_seq: np.ndarray) -> list:
        """
        Compare two full sequences using the sliding window approach.
        
        Args:
            user_seq: Full user pose recording.
            reference_seq: Full reference pose recording.
            
        Returns:
            list: A chronological list of similarity scores for each evaluated window.
        """
        user_seq = np.array(user_seq)
        reference_seq = np.array(reference_seq)
        
        num_frames = user_seq.shape[0]
        scores = []
        
        # Sliding window approach 
        for i in range(0, max(1, num_frames - self.window_size + 1)):
            end_idx = i + self.window_size
            user_window = user_seq[i:end_idx]
            
            # Assuming sequences are generally time-aligned (with local speed variations handled by DTW)
            # We match corresponding windows in time.
            ref_end_idx = min(reference_seq.shape[0], end_idx)
            ref_window = reference_seq[i:ref_end_idx]
            
            if len(user_window) == 0 or len(ref_window) == 0:
                continue
                
            score = self.compare_realtime(user_window, ref_window)
            scores.append(score)
            
        return scores
