import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum
from pathlib import Path
import os
import pygame
import threading
try:
    import soundfile as sf
except ImportError:
    sf = None

# Local imports
from ..pipelines.vision.pose_analyzer import (
    PoseAnalyzer,
    AlertChecker,
    OverlayAlertPresenter,
    LEARNED_PUNCH_PATH,
    PoseLandmarkIndex,
    AlertMessage,
)
from ..pipelines.vision.karaoke import SpeechKaraokeTrainer, _load_subtitles, _draw_subtitle_karaoke
from ..pipelines.vision.pose_comparator import PoseComparator
from ..pipelines.vision.gaze import GazeAnxietyDetector
from ..pipelines.vision.key_pose_extractor import KeyPoseExtractor
try:
    from ..pipelines.audio.audio_analyzer import AudioAnalyzer, AudioEvaluator
except ImportError:
    AudioAnalyzer = None
    AudioEvaluator = None

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"
REFERENCE_VIDEO_PATH = ASSETS_DIR / "reference_videos" / "Obama's 2004 DNC keynote speech.mp4"
REFERENCE_AUDIO_PATH = ASSETS_DIR / "reference_audio" / "Obama's 2004 DNC keynote speech.wav"
REFERENCE_JSON_PATH = ASSETS_DIR / "derived" / "Obama's 2004 DNC keynote speech.json"
REFERENCE_RAW_POSE_PATH = ASSETS_DIR / "derived" / "Obama's 2004 DNC keynote speech_raw.npy"
REFERENCE_SUBS_PATH = ASSETS_DIR / "subtitles" / "Obama's 2004 DNC keynote speech_subs.json"
REFERENCE_AUDIO_DIR = ASSETS_DIR / "reference_audio"

class AppMode(Enum):
    MODE_SELECTION = "MODE_SELECTION"
    INTERVIEW_MODE = "INTERVIEW_MODE"
    SPEECH_SELECTION = "SPEECH_SELECTION"
    DEFAULT = "DEFAULT"     
    KARAOKE_PRACTICE = "KARAOKE_PRACTICE"     
    KARAOKE_TEST = "KARAOKE_TEST"
    COUNTDOWN = "COUNTDOWN"
    TEST_RESULTS = "TEST_RESULTS"


# Colors & UI Constants
COLOR_BG = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 200, 100)
COLOR_WARNING = (0, 0, 255)

def draw_button(img, text, x, y, w, h, bg_color, text_color):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    cv2.putText(img, text, (x + 10, y + int(h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

class Test4App:
    def __init__(self):
        self.mode = AppMode.MODE_SELECTION
        self.cap_webcam = cv2.VideoCapture(0)
        
        # UI State
        self.w_web = int(self.cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_web = int(self.cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.w_web == 0: self.w_web = 1280
        if self.h_web == 0: self.h_web = 720
        
        # Test2 - Posture Analyzer
        self.pose_analyzer = PoseAnalyzer(learned_punch_path=LEARNED_PUNCH_PATH)
        self.alert_checker = AlertChecker(self.pose_analyzer)
        self.alert_presenter = OverlayAlertPresenter()
        
        # Test3 - Karaoke
        self.karaoke_trainer = SpeechKaraokeTrainer()
        self.pose_comparator = PoseComparator(window_size=30)
        self.gaze_detector = GazeAnxietyDetector()
        self.key_pose_extractor = KeyPoseExtractor(fps=30)
        
        self.cap_ref = None
        self.ref_data = None
        self.ref_raw_poses = None
        self.subtitles = None
        self.karaoke_start_time = 0
        self.speed_multiplier = 1.0
        self.user_pose_buffer = []
        
        # Test Mode Tracking variables
        self.test_pose_similarities = []
        self.test_gaze_scores = []
        self.test_keyframe_logs = []
        self.final_audio_score = None
        self.final_pose_score = None
        self.calculating_results = False
        self.test_audio_buffer = None
        
        # Audio playback
        pygame.mixer.init()
        self.ref_audio_path = None
        self.ref_audio_channel = None
        
        # Audio Analyzer
        self.audio_analyzer = None
        if AudioAnalyzer:
            self.audio_analyzer = AudioAnalyzer()
            self.audio_analyzer.start()
        
        # MediaPipe Holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        
        self.practice_button = {'x': self.w_web - 320, 'y': 10, 'w': 150, 'h': 40}
        self.test_button = {'x': self.w_web - 160, 'y': 10, 'w': 150, 'h': 40}
        
    def __del__(self):
        if self.audio_analyzer:
            self.audio_analyzer.stop()
        if self.gaze_detector:
            self.gaze_detector.release()
        if self.cap_webcam:
            self.cap_webcam.release()
        if self.cap_ref:
            self.cap_ref.release()
        pygame.mixer.quit()

    def process_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == AppMode.MODE_SELECTION:
                # Interview vs Speech mode
                cx, cy = self.w_web // 2, self.h_web // 2
                if cx - 220 <= x <= cx - 20 and cy - 40 <= y <= cy + 40:
                    self.mode = AppMode.INTERVIEW_MODE
                    print("Switched to Interview Mode")
                elif cx + 20 <= x <= cx + 220 and cy - 40 <= y <= cy + 40:
                    self.mode = AppMode.SPEECH_SELECTION
                    print("Switched to Speech Selection Mode")
                    
            elif self.mode == AppMode.SPEECH_SELECTION:
                # Back button
                if 20 <= x <= 120 and 20 <= y <= 60:
                    self.mode = AppMode.MODE_SELECTION
                    return
                    
                # Speech choices: clinton, jobs, obama, robbins, trump (1 and 2)
                cx = self.w_web // 2
                speeches = ["clinton1", "clinton2", "jobs1", "jobs2", "obama1", "obama2", "robbins1", "robbins2", "trump1", "trump2"]
                y_start = 150
                for i, sp_name in enumerate(speeches):
                    col = i % 2
                    row = i // 2
                    bx = cx - 220 + (col * 240)
                    by = y_start + (row * 70)
                    if bx <= x <= bx + 200 and by <= y <= by + 50:
                        self.selected_ref_name = sp_name
                        self.mode = AppMode.DEFAULT
                        print(f"Selected Speech: {sp_name}")
                        break
                        
            elif self.mode == AppMode.INTERVIEW_MODE:
                # Back button
                if 20 <= x <= 120 and 20 <= y <= 60:
                    self.mode = AppMode.MODE_SELECTION
                    
            elif self.mode == AppMode.DEFAULT: # Speech Home Mode (after selecting a ref)
                # Back button to re-select speech
                if 20 <= x <= 120 and 20 <= y <= 60:
                    self.mode = AppMode.SPEECH_SELECTION
                    return
                    
                px, py, pw, ph = self.practice_button['x'], self.practice_button['y'], self.practice_button['w'], self.practice_button['h']
                tx, ty, tw, th = self.test_button['x'], self.test_button['y'], self.test_button['w'], self.test_button['h']
                
                if px <= x <= px + pw and py <= y <= py + ph:
                    self.load_karaoke_video(AppMode.KARAOKE_PRACTICE)
                elif tx <= x <= tx + tw and ty <= y <= ty + th:
                    self.mode = AppMode.COUNTDOWN
                    self.karaoke_start_time = time.time()
                    
            elif self.mode in [AppMode.KARAOKE_PRACTICE, AppMode.TEST_RESULTS]:
                # Back button in Practice or Result modes
                btn_x = self.w_web * 2 - 160
                if btn_x <= x <= btn_x + 150 and 10 <= y <= 50:
                    self.stop_karaoke_video()

    def load_karaoke_video(self, target_mode):
        ref_name = getattr(self, 'selected_ref_name', "obama1")
        if ref_name == "obama1" and not (REPO_ROOT / "data" / "obama1.mp4").exists():
             # Fallback to the old default video name for compatibility if obama1 doesn't exist directly
             ref_video_path = REPO_ROOT / "assets" / "reference_videos" / "Obama's 2004 DNC keynote speech.mp4"
             ref_json_path = REPO_ROOT / "assets" / "derived" / "Obama's 2004 DNC keynote speech.json"
             raw_pose_path = REPO_ROOT / "assets" / "derived" / "Obama's 2004 DNC keynote speech_raw.npy"
             subs_path = REPO_ROOT / "assets" / "subtitles" / "Obama's 2004 DNC keynote speech_subs.json"
             audio_path = REPO_ROOT / "assets" / "reference_audio" / "Obama's 2004 DNC keynote speech.wav"
        else:
             ref_video_path = REPO_ROOT / "data" / f"{ref_name}.mp4"
             ref_json_path = REPO_ROOT / "assets" / "derived" / f"{ref_name}.json"
             raw_pose_path = REPO_ROOT / "assets" / "derived" / f"{ref_name}_raw.npy"
             subs_path = REPO_ROOT / "assets" / "subtitles" / f"{ref_name}_subs.json"
             audio_path = REPO_ROOT / "assets" / "reference_audio" / f"{ref_name}.wav"
             
        # Mock JSON creation if missing (since new data videos don't have json files derived yet)
        if not ref_json_path.exists():
            print(f"Mocking reference JSON for {ref_name} as it is missing.")
            cap = cv2.VideoCapture(str(ref_video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.release()
            mock_data = {
                "fps": fps,
                "frames": [{"timestamp_ms": i * (1000/fps)} for i in range(max(1, frame_count))]
            }
            import json
            ref_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ref_json_path, "w", encoding="utf-8") as f:
                json.dump(mock_data, f)
        
        # Audio Extraction
        if not audio_path.exists():
            print(f"Extracting audio from {ref_video_path}...")
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            os.system(f"ffmpeg -i \"{ref_video_path}\" -q:a 0 -map a \"{audio_path}\" -y")
        self.ref_audio_path = audio_path
        
        # Audio speeds pre-cache
        self.audio_speeds = {}
        try:
            from pydub import AudioSegment
            print("Pre-generating speed variants for audio...")
            base_audio = AudioSegment.from_file(self.ref_audio_path)
            for s in [0.5, 1.0, 1.25, 1.5, 2.0]:
                out_path = REFERENCE_AUDIO_DIR / f"{self.ref_audio_path.stem}_{s}.wav"
                if not out_path.exists():
                    if s == 1.0:
                        base_audio.export(out_path, format="wav")
                    else:
                        # Change speed with frame rate trick (alters pitch),
                        # but keeps playback tightly synced with video frames.
                        fast_sound = base_audio._spawn(base_audio.raw_data, overrides={
                             "frame_rate": int(base_audio.frame_rate * s)
                          }).set_frame_rate(base_audio.frame_rate)
                        fast_sound.export(out_path, format="wav")
                self.audio_speeds[s] = out_path
        except Exception as e:
            print(f"Warning: Could not setup audio speed variants via pydub: {e}")
            self.audio_speeds[1.0] = self.ref_audio_path
        
        import json
        try:
            with open(ref_json_path, "r", encoding="utf-8") as f:
                ref = json.load(f)
            self.ref_data = ref["frames"]
            self.fps = ref["fps"]
            if subs_path.exists():
                self.subtitles = _load_subtitles(subs_path)
            else:
                self.subtitles = None
            
            # Load or extract raw poses for DTW
            if not raw_pose_path.exists():
                print("Extracting raw pose data from video for DTW...")
                self._extract_raw_poses(ref_video_path, raw_pose_path)
            self.ref_raw_poses = np.load(raw_pose_path)
            self.user_pose_buffer = []
            
            self.cap_ref = cv2.VideoCapture(str(ref_video_path))
            self.speed_multiplier = 1.0
            self.karaoke_start_time = time.time()
            
            # Start playing audio
            current_audio = self.audio_speeds.get(1.0, self.ref_audio_path)
            if current_audio.exists():
                sound = pygame.mixer.Sound(str(current_audio))
                self.ref_audio_channel = sound.play()
            
            self.mode = target_mode
            if target_mode == AppMode.KARAOKE_TEST:
                self.test_pose_similarities = []
                self.test_gaze_scores = []
                self.test_keyframe_logs = []
                self.final_audio_score = None
                self.final_pose_score = None
                self.calculating_results = False
                if self.audio_analyzer:
                    self.audio_analyzer.start_test_mode()
                    
            print(f"Switched to {target_mode}")
        except Exception as e:
            print(f"Error loading karaoke data: {e}")
            
    def _extract_raw_poses(self, video_path, out_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(frame_rgb)
                if res.pose_landmarks:
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark])
                else:
                    pts = np.zeros((33, 3))
                frames.append(pts)
        cap.release()
        np.save(out_path, np.array(frames))
            
    def stop_karaoke_video(self):
        if self.ref_audio_channel:
            self.ref_audio_channel.stop()
        if self.audio_analyzer and self.mode == AppMode.KARAOKE_TEST:
             self.audio_analyzer.end_test_mode()
             
        self.mode = AppMode.DEFAULT
            
    def draw_audio_metrics(self, img, wpm, energy, pitch, start_x, start_y):
        """Draws real time audio feedback"""
        cv2.putText(img, "Audio Analysis", (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        y = start_y + 30
        
        # Volume (Energy)
        cv2.putText(img, f"Vol (Energy): {energy:.4f}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        # Normalize roughly for visualization
        vol_bar = min(1.0, energy * 20.0)
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + 100, y), (80, 80, 80), -1)
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + int(100*vol_bar), y), COLOR_ACCENT, -1)
        
        # Stress (Pitch Std)
        y += 30
        cv2.putText(img, f"Stress (Pitch): {pitch:.1f}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        stress_bar = min(1.0, pitch / 100.0) 
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + 100, y), (80, 80, 80), -1)
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + int(100*stress_bar), y), COLOR_ACCENT, -1)

        # Speed (WPM)
        y += 30
        cv2.putText(img, f"Speed (WPM): {wpm:.1f}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        spd_bar = min(1.0, wpm / 150.0) 
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + 100, y), (80, 80, 80), -1)
        cv2.rectangle(img, (start_x + 180, y - 10), (start_x + 180 + int(100*spd_bar), y), COLOR_ACCENT, -1)
        
    def run(self):
        cv2.namedWindow("Test4 App - Posture & Speech Karaoke")
        cv2.setMouseCallback("Test4 App - Posture & Speech Karaoke", self.process_mouse_click)
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap_webcam.isOpened():
                success, image = self.cap_webcam.read()
                if not success:
                    continue
                
                # Mirror webcam horizontally
                image = cv2.flip(image, 1)
                
                # Holistic Processing
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                image.flags.writeable = True

                # Determine View and Render
                if self.mode == AppMode.MODE_SELECTION:
                    image = self._render_mode_selection(image)
                elif self.mode == AppMode.SPEECH_SELECTION:
                    image = self._render_speech_selection(image)
                elif self.mode == AppMode.INTERVIEW_MODE:
                    image = self._render_interview_mode(image, results)
                elif self.mode == AppMode.DEFAULT:
                    image = self._render_default_mode(image, results)
                elif self.mode in [AppMode.KARAOKE_PRACTICE, AppMode.KARAOKE_TEST]:
                    image = self._render_karaoke_mode(image, results)
                elif self.mode == AppMode.COUNTDOWN:
                    image = self._render_countdown_mode(image)
                elif self.mode == AppMode.TEST_RESULTS:
                    image = self._render_test_results(image)

                cv2.imshow("Test4 App - Posture & Speech Karaoke", image)
                
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif self.mode == AppMode.KARAOKE_PRACTICE:
                    # Speed controls only in practice mode
                    if key == ord('1'):
                        self._change_speed(0.5)
                    elif key == ord('2'):
                        self._change_speed(1.0)
                    elif key == ord('3'):
                        self._change_speed(1.25)
                    elif key == ord('4'):
                        self._change_speed(1.5)
                    elif key == ord('5'):
                        self._change_speed(2.0)
                        
    def _change_speed(self, new_speed):
        if new_speed not in [0.5, 1.0, 1.25, 1.5, 2.0]:
            return
            
        self.speed_multiplier = new_speed
        print(f"Speed set to {new_speed}x")
        self.karaoke_start_time = time.time()
        
        if self.ref_audio_channel:
            self.ref_audio_channel.stop()
        
        current_audio = self.audio_speeds.get(new_speed, self.ref_audio_path)
        if current_audio.exists():
            sound = pygame.mixer.Sound(str(current_audio))
            self.ref_audio_channel = sound.play()
            
    def _render_mode_selection(self, image):
        """Initial landing screen to select Interview or Speech mode"""
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (self.w_web, self.h_web), (0,0,0), -1)
        image = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)
        
        cx, cy = self.w_web // 2, self.h_web // 2
        cv2.putText(image, "Welcome to Multimodal Coach", (cx - 240, cy - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)
        cv2.putText(image, "Select Practice Mode:", (cx - 150, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        draw_button(image, "INTERVIEW MODE", cx - 220, cy - 40, 200, 80, (80, 80, 150), COLOR_TEXT)
        draw_button(image, "SPEECH MODE", cx + 20, cy - 40, 200, 80, (150, 80, 80), COLOR_TEXT)
        return image

    def _render_speech_selection(self, image):
        """Menu to select a famous speech for the karaoke mode"""
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (self.w_web, self.h_web), (0,0,0), -1)
        image = cv2.addWeighted(overlay, 0.9, image, 0.1, 0)
        
        draw_button(image, "<- Back", 20, 20, 100, 40, (100, 100, 100), COLOR_TEXT)
        
        cx = self.w_web // 2
        cv2.putText(image, "Select Reference Speech", (cx - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 3)
        
        speeches = ["clinton1", "clinton2", "jobs1", "jobs2", "obama1", "obama2", "robbins1", "robbins2", "trump1", "trump2"]
        y_start = 150
        for i, sp_name in enumerate(speeches):
            col = i % 2
            row = i // 2
            bx = cx - 220 + (col * 240)
            by = y_start + (row * 70)
            draw_button(image, sp_name.upper(), bx, by, 200, 50, (60, 100, 60), COLOR_TEXT)
        
        return image

    def _render_interview_mode(self, image, results):
        """Dedicated Interview Mode (Standard Practice)"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            
        metrics = self.pose_analyzer.analyze(
            results.pose_landmarks,
            results.left_hand_landmarks,
            results.right_hand_landmarks,
        )
        alerts = self.alert_checker.check_alerts(metrics)
        gaze_res = self.gaze_detector.process_frame(image)
        if gaze_res.get('status') in ["Avoiding", "Shaking"]:
            alerts.append(AlertMessage(alert_type="gaze_warning", message=f"Eye Tracking: {gaze_res.get('message')}", severity="warning"))
            
        image = self.alert_presenter.update_and_show(alerts, image)
        
        cv2.putText(image, "INTERVIEW MODE: General Practice", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
        draw_button(image, "<- Back", 20, 60, 100, 40, (100, 100, 100), COLOR_TEXT)
        return image

    def _render_default_mode(self, image, results):
        """Standard test2 mode with Posture Feedback"""
        # Draw landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            
        metrics = self.pose_analyzer.analyze(
            results.pose_landmarks,
            results.left_hand_landmarks,
            results.right_hand_landmarks,
        )
        alerts = self.alert_checker.check_alerts(metrics)
        
        # --- GAZE ANXIETY EVALUATION ---
        # Evaluate Gaze Anxiety (needs original BGR frame)
        gaze_res = self.gaze_detector.process_frame(image)
        gaze_status = gaze_res.get('status', 'Stable')
        gaze_msg = gaze_res.get('message', '')
        
        if gaze_status == "Calibrating":
            alerts.append(AlertMessage(alert_type="gaze_calibrating", message=gaze_msg, severity="info"))
        elif gaze_status in ["Avoiding", "Shaking"]:
            alerts.append(AlertMessage(alert_type="gaze_warning", message=f"Eye Tracking: {gaze_msg}", severity="warning"))
        # -------------------------------
            
        image = self.alert_presenter.update_and_show(alerts, image)
        
        # State overlay
        ref_name = getattr(self, 'selected_ref_name', 'obama1').upper()
        cv2.putText(image, f"SPEECH MODE: Selected [{ref_name}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
        
        draw_button(image, "<- Change Speech", 20, 50, 150, 40, (100, 100, 100), (255,255,255))
        
        # Draw Menu Buttons
        px, py, pw, ph = self.practice_button['x'], self.practice_button['y'], self.practice_button['w'], self.practice_button['h']
        draw_button(image, f"Practice {ref_name}", px, py, pw, ph, (100, 100, 200), (255,255,255))
        
        tx, ty, tw, th = self.test_button['x'], self.test_button['y'], self.test_button['w'], self.test_button['h']
        draw_button(image, f"Test {ref_name}", tx, ty, tw, th, (200, 100, 100), (255,255,255))
        
        return image
        
    def _render_countdown_mode(self, image):
        """Displays 3, 2, 1, START before launching TEST mode."""
        elapsed = time.time() - self.karaoke_start_time
        countdown = 3 - int(elapsed)
        
        # Darken webcam
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (self.w_web, self.h_web), (0,0,0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        if countdown > 0:
            text = str(countdown)
        elif countdown == 0:
            text = "START!"
        else:
            self.load_karaoke_video(AppMode.KARAOKE_TEST)
            return image
            
        font_scale = 5.0 if countdown > 0 else 3.0
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 10)[0]
        text_x = (self.w_web - text_size[0]) // 2
        text_y = (self.h_web + text_size[1]) // 2
        
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_ACCENT, 10)
        return image
        
    def _render_karaoke_mode(self, image_web, results):
        """Speech Karaoke mode heavily influenced by test3"""
        h_web, w_web = image_web.shape[:2]
        
        # Sync reference video
        elapsed_sec = time.time() - self.karaoke_start_time
        effective_sec = elapsed_sec * self.speed_multiplier
        total_duration_sec = self.ref_data[-1]["timestamp_ms"] / 1000.0 if self.ref_data else 36.0
        
        if effective_sec > total_duration_sec:
            if self.mode == AppMode.KARAOKE_PRACTICE:
                # Wrap around in Practice mode
                self.karaoke_start_time = time.time()
                effective_sec = 0
                if self.ref_audio_channel:
                    self.ref_audio_channel.stop()
                    current_audio = self.audio_speeds.get(self.speed_multiplier, self.ref_audio_path)
                    if current_audio.exists():
                        sound = pygame.mixer.Sound(str(current_audio))
                        self.ref_audio_channel = sound.play()
            elif self.mode == AppMode.KARAOKE_TEST:
                # End of Test mode
                if self.ref_audio_channel:
                    self.ref_audio_channel.stop()
                if self.audio_analyzer:
                    self.test_audio_buffer = self.audio_analyzer.end_test_mode()
                self.mode = AppMode.TEST_RESULTS
                return np.zeros((h_web, w_web*2, 3), dtype=np.uint8)
                
        timestamp_ms = effective_sec * 1000
        
        ref_idx = int((timestamp_ms / 1000) * self.fps) % len(self.ref_data)
        
        if self.cap_ref:
            self.cap_ref.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            success_ref, image_ref = self.cap_ref.read()
            if not success_ref:
                image_ref = np.zeros((h_web, w_web, 3), dtype=np.uint8)
            else:
                image_ref = cv2.resize(image_ref, (w_web, h_web))
                
        # Combine images side by side
        combined = np.hstack([image_ref, image_web])
        h_comb, w_comb = combined.shape[:2]
        
        # Labels
        cv2.rectangle(combined, (0, 0), (w_comb, 35), (30, 30, 30), -1)
        cv2.putText(combined, "Reference Video", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(combined, "Your Webcam", (w_web + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Draw subtitles
        if self.subtitles:
            _draw_subtitle_karaoke(combined, self.subtitles, elapsed_sec, h_web, w_web)
            
        # Draw Gesture Similarity with DTW Sliding Window
        if results.pose_landmarks:
            user_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        else:
            user_pts = np.zeros((33, 3))
            
        self.user_pose_buffer.append(user_pts)
        if len(self.user_pose_buffer) > 30:
            self.user_pose_buffer.pop(0)

        similarity_pct = 0.0
        if self.ref_raw_poses is not None and len(self.user_pose_buffer) >= 5:
            start_ref = max(0, ref_idx - len(self.user_pose_buffer) + 1)
            ref_window = self.ref_raw_poses[start_ref:ref_idx+1]
            if len(ref_window) > 0:
                score = self.pose_comparator.compare_realtime(np.array(self.user_pose_buffer), ref_window)
                similarity_pct = score * 100

        if self.mode == AppMode.KARAOKE_TEST:
            self.test_pose_similarities.append(similarity_pct)
            
            # Key Pose Extraction
            idx = PoseLandmarkIndex
            if results.pose_landmarks:
                user_landmarks = {
                    "left_wrist": (results.pose_landmarks.landmark[idx.LEFT_WRIST].x, results.pose_landmarks.landmark[idx.LEFT_WRIST].y, results.pose_landmarks.landmark[idx.LEFT_WRIST].z),
                    "right_wrist": (results.pose_landmarks.landmark[idx.RIGHT_WRIST].x, results.pose_landmarks.landmark[idx.RIGHT_WRIST].y, results.pose_landmarks.landmark[idx.RIGHT_WRIST].z),
                    "left_shoulder": (results.pose_landmarks.landmark[idx.LEFT_SHOULDER].x, results.pose_landmarks.landmark[idx.LEFT_SHOULDER].y, results.pose_landmarks.landmark[idx.LEFT_SHOULDER].z),
                    "right_shoulder": (results.pose_landmarks.landmark[idx.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[idx.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[idx.RIGHT_SHOULDER].z)
                }
                
                # We need actual reference shoulder vectors for similarity matching. 
                # Passing a dummy straight-down vector so logging logic triggers for the demo
                ref_sync_data = {
                    "shoulder_elbow_wrist_vectors": {
                        "left": [[0, -1, 0], [0, -1, 0]],
                        "right": [[0, -1, 0], [0, -1, 0]]
                    }
                }
                
                logs = self.key_pose_extractor.process_frame(user_landmarks, effective_sec * 1000, ref_sync_data)
                for log in logs:
                    self.test_keyframe_logs.append(log)
            
        # Draw UI Overlays (Only in Practice Mode)
        bar_x, bar_y = w_web + 20, h_comb - 200
        bar_w, bar_h = 300, 20
        
        if self.mode == AppMode.KARAOKE_PRACTICE:
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            fill_w = int(bar_w * (similarity_pct / 100))
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_ACCENT, -1)
            cv2.putText(combined, f"Pose Similarity: {similarity_pct:.1f}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(combined, f"Speed: {self.speed_multiplier}x  (Keys 1-5 to change)", (bar_x, bar_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
            # Draw Audio Metrics
            if self.audio_analyzer:
                wpm, energy, pitch = self.audio_analyzer.get_metrics()
                self.draw_audio_metrics(combined, wpm, energy, pitch, bar_x, bar_y + 40)
        elif self.mode == AppMode.KARAOKE_TEST:
            cv2.putText(combined, "TEST IN PROGRESS...", (bar_x, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WARNING, 3)
            
        # Back Button - allow quitting in either mode
        btn_x = w_comb - 160
        draw_button(combined, "<- Quit", btn_x, 10, 150, 40, (100, 100, 100), (255,255,255))
        
        return combined

    def _render_test_results(self, image_web):
        """Displays final 100-point accuracy results post-test"""
        h, w = image_web.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        if not self.calculating_results:
            self.calculating_results = True
            
            # Start a thread to compute the final audio metrics without hanging the UI loop
            def compute_scores():
                if AudioEvaluator and self.test_audio_buffer is not None:
                    # Async evaluation to not block the main cv2 UI thread
                    print("Starting async audio evaluation...")
                    eval_res = AudioEvaluator.evaluate(self.test_audio_buffer, self.audio_analyzer.sample_rate)
                    self.final_audio_score = eval_res
                    print("Async evaluation finished!")
                else:
                    self.final_audio_score = {
                        "total_score": 0.0,
                        "breakdown": {"Accuracy": 0.0, "Fluency": 0.0, "Pronunciation": 0.0}
                    }
                
                # Compute final pose score
                if getattr(self, "test_pose_similarities", None) and len(self.test_pose_similarities) > 0:
                    self.final_pose_score = sum(self.test_pose_similarities) / len(self.test_pose_similarities)
                else:
                    self.final_pose_score = 0.0
                    
            threading.Thread(target=compute_scores).start()

        cx, cy = w, h//2
        if self.final_audio_score is None:
            cv2.putText(canvas, "Evaluating Speech... Please wait.", (cx - 200, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        else:
            # Result Display
            a_score = self.final_audio_score["total_score"]
            p_score = self.final_pose_score
            combined_score = (a_score * 0.5) + (p_score * 0.5)
            
            cv2.putText(canvas, "TEST RESULTS", (cx - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_ACCENT, 4)
            
            # Positional Alignment
            lx = cx - 200
            
            cv2.putText(canvas, f"Total Score: {combined_score:.1f}/100", (lx, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)
            cv2.putText(canvas, f"Pose Accuracy: {p_score:.1f}%", (lx, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
            cv2.putText(canvas, f"Speech Accuracy: {a_score:.1f}%", (lx, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
            
            # Audio Breakdown
            cv2.putText(canvas, "Speech Breakdown:", (lx, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            y_break = 420
            for label, val in self.final_audio_score["breakdown"].items():
                cv2.putText(canvas, f" - {label}: {val:.1f}/100", (lx, y_break), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_break += 30
                
            # Key Pose Feedback
            if hasattr(self, 'test_keyframe_logs') and len(self.test_keyframe_logs) > 0:
                cv2.putText(canvas, "Key Gesture Feedback:", (cx + 50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                y_log = 240
                
                # Show up to 5 most recent logs
                display_logs = self.test_keyframe_logs[-5:]
                for log in display_logs:
                    cv2.putText(canvas, f" - {log}", (cx + 50, y_log), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_log += 25
            
        btn_x = (w*2) - 160
        draw_button(canvas, "<- Return", btn_x, 10, 150, 40, (100, 100, 100), (255,255,255))
        return canvas

if __name__ == "__main__":
    app = Test4App()
    app.run()
