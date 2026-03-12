import unittest
import numpy as np
from multimodal_coach.pipelines.vision.pose_comparator import PoseComparator

class TestPoseComparator(unittest.TestCase):
    def setUp(self):
        self.comparator = PoseComparator(window_size=30)
        self.num_landmarks = 33
        
    def generate_dummy_pose(self, frames, offset=0.0, scale=1.0, base_pose=None):
        """Generates a dummy pose sequence of shape [frames, num_landmarks, 3]"""
        # Create a base skeleton that moves slightly over time
        if base_pose is None:
            base_pose = np.random.rand(self.num_landmarks, 3) 
        sequence = []
        for i in range(frames):
            # Add some linear movement and apply scale/offset
            frame = (base_pose + i * 0.01) * scale + offset
            sequence.append(frame)
        return np.array(sequence), base_pose

    def test_exact_match(self):
        # Two identical sequences should yield a score of 1.0
        seq1, _ = self.generate_dummy_pose(30)
        score = self.comparator.compare_realtime(seq1, seq1)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_scale_translation_invariance(self):
        # A sequence and its scaled/translated version should match perfectly 
        # because of the _preprocess L2 normalization and centering
        seq1, base_pose = self.generate_dummy_pose(30)
        seq2, _ = self.generate_dummy_pose(30, offset=100.0, scale=5.0, base_pose=base_pose)
        
        score = self.comparator.compare_realtime(seq1, seq2)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_speed_variation(self):
        # DTW should handle minor speed variations well
        # seq1 is normal speed (30 frames)
        seq1, _ = self.generate_dummy_pose(30)
        
        # seq2 is the same movement but slightly slower (takes 35 frames)
        # We simulate this by duplicating some frames
        seq2 = np.zeros((35, self.num_landmarks, 3))
        indices = np.linspace(0, 29, 35).astype(int)
        for i, idx in enumerate(indices):
            seq2[i] = seq1[idx]
            
        score = self.comparator.compare_realtime(seq1, seq2)
        self.assertGreater(score, 0.9)  # Should still be a very high match

    def test_different_poses(self):
        # Two completely different random poses should have a lower score
        seq1 = np.random.rand(30, self.num_landmarks, 3)
        seq2 = np.random.rand(30, self.num_landmarks, 3)
        
        score = self.comparator.compare_realtime(seq1, seq2)
        self.assertLess(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_sliding_window(self):
        # Test the compare_full_sequences which uses a sliding window
        user_seq, _ = self.generate_dummy_pose(60) # 60 frames
        # Make reference identical for the first 45 frames, then differ
        ref_seq = np.copy(user_seq)
        ref_seq[45:] = np.random.rand(15, self.num_landmarks, 3)
        
        scores = self.comparator.compare_full_sequences(user_seq, ref_seq)
        
        # We expect (60 - 30 + 1) = 31 windows
        self.assertEqual(len(scores), 31)
        
        # The first window (0:30) should be a perfect match
        self.assertAlmostEqual(scores[0], 1.0, places=4)
        
        # The last window (30:60) contains the randomised part, so score should drop
        self.assertLess(scores[-1], 1.0)

if __name__ == '__main__':
    unittest.main()
