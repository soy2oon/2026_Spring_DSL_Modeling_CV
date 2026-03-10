"""
Background Audio Analyzer for Speech Karaoke (test4.py)
"""
import threading
import time
import numpy as np
import librosa
import whisper
import queue
import re

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sd = None

# Using whisper base for faster evaluation
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading whisper model (try 'tiny' if base fails): {e}")
    whisper_model = None


class AudioAnalyzer:
    def __init__(self, sample_rate=16000, chunk_duration=4.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration # seconds to record per chunk
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        self.is_running = False
        self.thread = None
        
        # Latest metrics to be read by the main UI thread (Practice mode)
        self.latest_wpm = 0.0
        self.latest_energy = 0.0
        self.latest_pitch_std = 0.0
        
        # Test Mode properties
        self.is_test_mode = False
        self.test_audio_buffer = np.zeros(0, dtype=np.float32)
        
        # For recording
        self.audio_queue = queue.Queue()
        
    def start(self):
        if self.is_running or sd is None:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._recording_and_analysis_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def start_test_mode(self):
        self.is_test_mode = True
        self.test_audio_buffer = np.zeros(0, dtype=np.float32)
        
    def end_test_mode(self):
        self.is_test_mode = False
        final_buffer = self.test_audio_buffer.copy()
        return final_buffer
            
    def _audio_callback(self, indata, frames, time, status):
        """This is called continuously by sounddevice for every audio block"""
        if status:
            pass # ignore overruns
        # Flatten and put in queue
        self.audio_queue.put(indata.copy().flatten())

    def _recording_and_analysis_loop(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
            buffer = np.zeros(0, dtype=np.float32)
            
            while self.is_running:
                try:
                    # Collect data
                    data = self.audio_queue.get(timeout=0.5)
                    buffer = np.concatenate((buffer, data))
                    
                    if self.is_test_mode:
                        self.test_audio_buffer = np.concatenate((self.test_audio_buffer, data))
                    
                    # If we have gathered enough duration, run analysis (Live Feedback for Practice)
                    if len(buffer) >= self.chunk_samples:
                        # Take the latest chunk_samples
                        chunk_to_analyze = buffer[-self.chunk_samples:]
                        
                        # Remove half the chunk from buffer to have sliding window overlaps
                        buffer = buffer[self.chunk_samples//2:] 
                        
                        if not self.is_test_mode:  # Don't waste CPU on chunking during test mode
                            self._analyze_chunk(chunk_to_analyze)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio Analyzer Error: {e}")
                    
    def _analyze_chunk(self, y):
        """Runs librosa and Whisper on the audio chunk"""
        if len(y) == 0:
            return
            
        # 1. Energy (Volume)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        energy_mean = float(np.mean(rms))
        
        # 2. Pitch / Stress
        # Downsample for faster pitch tracking
        y_pitch = librosa.resample(y, orig_sr=self.sample_rate, target_sr=8000) if self.sample_rate > 8000 else y
        f0, _, _ = librosa.pyin(y_pitch, fmin=70, fmax=400, sr=8000, hop_length=1024)
        pitch_std = float(np.nanstd(f0)) if np.any(~np.isnan(f0)) else 0.0
        
        # 3. Whisper / Speed (WPM)
        wpm = 0.0
        if whisper_model is not None:
            # Whisper expects Float32 [-1, 1] which sounddevice provides
            # But the chunk is small, might not be enough context for accurate Whisper
            try:
                # we don't need highly accurate timestamps, just the text count
                result = whisper_model.transcribe(y, fp16=False, language="ko")
                text = result.get("text", "").strip()
                words = len(text.split())
                actual_duration = len(y) / self.sample_rate
                if actual_duration > 0:
                    wpm = words / (actual_duration / 60.0)
            except Exception:
                pass
                
        # Update shared state
        # Add a little smoothing
        self.latest_energy = 0.7 * self.latest_energy + 0.3 * energy_mean
        if pitch_std > 0:
            self.latest_pitch_std = 0.7 * self.latest_pitch_std + 0.3 * pitch_std
        if wpm > 0:
            self.latest_wpm = 0.5 * self.latest_wpm + 0.5 * wpm

    def get_metrics(self):
        """Returns (wpm, energy, pitch_std) to be displayed in UI."""
        return self.latest_wpm, self.latest_energy, self.latest_pitch_std


# ============================================================
# Final Evaluation Logic (from Copy_of_whisper_small.ipynb)
# ============================================================

def clip01_to_100(x):
    return float(np.clip(x, 0.0, 100.0))

class AudioEvaluator:
    """Wrapper class implementing the logic in Copy_of_whisper_small.ipynb"""
    
    TEMPO_MU = 125
    TEMPO_SIGMA = 40
    ALPHA = 0.03
    PITCH_LOW, PITCH_HIGH = 30, 65
    BETA = 0.04
    E_LOW, E_HIGH = 0.02, 0.05
    GAMMA_ENERGY = 30.0
    K_FILLER = 5
    PAUSE_MU = 0.78
    PAUSE_SIGMA = 0.10
    TOP_DB = 30
    LAM_FLUENCY = 0.55
    LAM_TEMPO = 0.20
    FLOOR_PAUSE = 30.0
    FLOOR_TEMPO = 20.0
    CALIB_P = 0.55
    CALIB_MIN = 0.0
    CALIB_MAX = 75.0

    @classmethod
    def evaluate(cls, y, sr):
        if len(y) == 0:
            return {"total_score": 0, "breakdown": {}}
            
        total_dur = len(y) / sr
        
        # ASR
        text = ""
        wpm = 0.0
        var_wpm = 0.0
        
        if whisper_model is not None:
            try:
                result = whisper_model.transcribe(y, language="ko", fp16=False)
                text = result.get("text", "").strip()
                words = max(1, len(text.split())) # prevent div by 0
                wpm = words / max(1e-9, (total_dur / 60.0))
                
                # Segment wpm variance
                seg_wpms = []
                for seg in result.get("segments", []):
                    start = float(seg.get("start", 0.0))
                    end   = float(seg.get("end", 0.0))
                    dur = max(1e-6, end - start)
                    swords = len(str(seg.get("text", "")).strip().split())
                    if swords > 0:
                        seg_wpms.append(swords / (dur / 60.0))
                var_wpm = float(np.std(seg_wpms)) if len(seg_wpms) >= 2 else 0.0
            except Exception as e:
                print(f"Evaluation Whisper Error: {e}")
                
        # Pitch / Energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        energy_std  = float(np.std(rms))

        # Downsample for faster pitch tracking using standard yin
        y_pitch = librosa.resample(y, orig_sr=sr, target_sr=8000) if sr > 8000 else y
        f0 = librosa.yin(y_pitch, fmin=70, fmax=400, sr=8000, hop_length=1024)
        
        # Mask out unvoiced/silent frames manually since yin doesn't return probabilities
        rms_pitch = librosa.feature.rms(y=y_pitch, frame_length=2048, hop_length=1024)[0]
        min_len = min(len(f0), len(rms_pitch))
        f0 = f0[:min_len]
        rms_pitch = rms_pitch[:min_len]
        
        f0_voiced = f0[rms_pitch > 0.01]
        pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        
        # Fillers
        fillers = ["어", "음", "그"]
        tokens = text.lower().split()
        filler_counts = 0
        for t in tokens:
            t_clean = re.sub(r"[^\w가-힣]", "", t)
            if t_clean in fillers:
                filler_counts += 1
                
        n_words = max(1, len(text.split()))
        fr = filler_counts / n_words
        s_fluency = clip01_to_100(100.0 * (1.0 - cls.K_FILLER * fr))
        
        # Tempo
        s_rate = clip01_to_100(100.0 * (1.0 - ((wpm - cls.TEMPO_MU) ** 2) / (cls.TEMPO_SIGMA ** 2)))
        s_stability = clip01_to_100(100.0 * np.exp(-cls.ALPHA * var_wpm))
        s_tempo = clip01_to_100(0.6 * s_rate + 0.4 * s_stability)
        
        # Pitch Score
        s_pitch = 0.0
        if not np.isnan(pitch_std):
            if pitch_std < cls.PITCH_LOW:
                s_pitch = clip01_to_100(100.0 * (pitch_std / cls.PITCH_LOW))
            elif cls.PITCH_LOW <= pitch_std <= cls.PITCH_HIGH:
                s_pitch = 100.0
            else:
                s_pitch = clip01_to_100(100.0 * np.exp(-cls.BETA * (pitch_std - cls.PITCH_HIGH)))
                
        # Energy Score
        s_energy = 0.0
        if energy_std < cls.E_LOW:
            s_energy = clip01_to_100(100.0 * (energy_std / cls.E_LOW))
        elif cls.E_LOW <= energy_std <= cls.E_HIGH:
            s_energy = 100.0
        else:
            s_energy = clip01_to_100(100.0 * np.exp(-cls.GAMMA_ENERGY * (energy_std - cls.E_HIGH)))

        # Pause
        intervals = librosa.effects.split(y, top_db=cls.TOP_DB, frame_length=2048, hop_length=1024)
        speech_time = sum((end - start) for start, end in intervals) / sr
        if total_dur <= 1e-9:
            s_pause = cls.FLOOR_PAUSE
        else:
            pr = speech_time / total_dur
            raw_p = 100.0 * (1.0 - ((pr - cls.PAUSE_MU) ** 2) / (cls.PAUSE_SIGMA ** 2))
            s_pause = max(cls.FLOOR_PAUSE, clip01_to_100(raw_p))
            
        # Total
        tempo_g = max(cls.FLOOR_TEMPO, s_tempo)
        pause_g = max(cls.FLOOR_PAUSE, s_pause)
        base = float(np.mean([tempo_g, s_pitch, s_energy, pause_g]))
        
        mult_f = np.clip(1.0 - cls.LAM_FLUENCY * (1.0 - s_fluency/100.0), 0.0, 1.0)
        mult_t = np.clip(1.0 - cls.LAM_TEMPO * (1.0 - tempo_g/100.0), 0.0, 1.0)
        
        gated = base * mult_f * mult_t
        
        # Calibrate
        x = (gated - cls.CALIB_MIN) / max(1e-9, (cls.CALIB_MAX - cls.CALIB_MIN))
        x = np.clip(x, 0.0, 1.0)
        total_final = float(100.0 * (x ** cls.CALIB_P))
        
        return {
            "total_score": total_final,
            "breakdown": {
                "Tempo": s_tempo,
                "Pitch (Stress)": s_pitch,
                "Energy (Volume)": s_energy,
                "Fluency": s_fluency,
                "Pauses": s_pause
            }
        }
