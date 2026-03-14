import time
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sounddevice as sd
import torch
import opensmile

from silero_vad import load_silero_vad, get_speech_timestamps


# ============================================================
# Config
# ============================================================
SR = 16000
BLOCK_SEC = 0.50
BLOCK_SIZE = int(SR * BLOCK_SEC)

CALIBRATION_SEC = 7.0
POPUP_DURATION_MS = 3000
ALERT_COOLDOWN_SEC = 3.0

# realtime thresholds (personal baseline z-score)
Z_HIGH = 2.2
Z_LOW = 2.2

# pause
MICRO_PAUSE_SEC = 0.30
K_PAUSE = 2.0
FALLBACK_TAU_PAUSE = 1.0

# silero speech decision on each chunk
MIN_SPEECH_RATIO = 0.20  # chunk 내 speech 비율이 20% 넘으면 speech chunk로 판단


# ============================================================
# Popup manager
# ============================================================
class PopupManager:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.window = None
        self.label = None
        self.hide_job = None
        self.last_alert_time = {}
        self.current_message = None

    def show(self, message: str, key: str):
        now = time.time()
        last_t = self.last_alert_time.get(key, 0.0)
        if now - last_t < ALERT_COOLDOWN_SEC:
            return
        self.last_alert_time[key] = now

        if self.window is None or not self.window.winfo_exists():
            self.window = tk.Toplevel(self.root)
            self.window.title("Speech Feedback")
            self.window.attributes("-topmost", True)
            self.window.configure(bg="red")

            self.label = tk.Label(
                self.window,
                text=message,
                font=("Helvetica", 16, "bold"),
                fg="white",
                bg="red",
                padx=20,
                pady=20,
            )
            self.label.pack()

            self.window.update_idletasks()
            w = self.window.winfo_width()
            h = self.window.winfo_height()
            x = (self.window.winfo_screenwidth() - w) // 2
            y = 80
            self.window.geometry(f"+{x}+{y}")
        else:
            self.label.config(text=message)
            self.window.deiconify()
            self.window.lift()

        if self.hide_job is not None:
            self.root.after_cancel(self.hide_job)

        self.hide_job = self.root.after(POPUP_DURATION_MS, self.hide)

    def hide(self):
        if self.window is not None and self.window.winfo_exists():
            self.window.withdraw()


# ============================================================
# Monitor state
# ============================================================
@dataclass
class MonitorState:
    processed_sec: float = 0.0
    calibration_done: bool = False

    # baseline
    pitch_vals_cal: list = field(default_factory=list)
    energy_vals_cal: list = field(default_factory=list)
    pause_vals_cal: list = field(default_factory=list)

    mu_pitch: float = np.nan
    sd_pitch: float = np.nan
    mu_energy: float = np.nan
    sd_energy: float = np.nan
    tau_pause: float = FALLBACK_TAU_PAUSE

    # running pause
    current_silence_sec: float = 0.0

    # eval stats
    eval_pitch_total: int = 0
    eval_pitch_viol: int = 0

    eval_energy_total: int = 0
    eval_energy_viol: int = 0

    eval_pause_excess_sec: float = 0.0
    eval_time_sec: float = 0.0

    # logging
    started: bool = False


# ============================================================
# Main monitor
# ============================================================
class RealtimeSpeechMonitor:
    def __init__(self, popup_manager: PopupManager):
        self.popup = popup_manager
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.state = MonitorState()

        # Silero
        self.vad_model = load_silero_vad()

        # openSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        self.pitch_col = None
        self.energy_col = None

        self.stream = None

    # -------------------------
    # audio callback
    # -------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        mono = indata[:, 0].copy()
        self.audio_queue.put(mono)

    # -------------------------
    # silero speech decision
    # -------------------------
    def is_speech_chunk(self, chunk: np.ndarray) -> bool:
        wav = torch.from_numpy(chunk.astype(np.float32))
        ts = get_speech_timestamps(wav, self.vad_model, sampling_rate=SR)
        if len(ts) == 0:
            return False

        speech_samples = sum(max(0, seg["end"] - seg["start"]) for seg in ts)
        ratio = speech_samples / max(1, len(chunk))
        return ratio >= MIN_SPEECH_RATIO

    # -------------------------
    # opensmile feature extraction
    # -------------------------
    def extract_pitch_energy(self, chunk: np.ndarray):
        df = self.smile.process_signal(chunk, SR)
        if df is None or len(df) == 0:
            return np.nan, np.nan

        if self.pitch_col is None or self.energy_col is None:
            cols = list(df.columns)
            pitch_candidates = [c for c in cols if "f0semitonefrom27.5hz" in c.lower()]
            energy_candidates = [c for c in cols if "loudness" in c.lower()]

            self.pitch_col = pitch_candidates[0] if pitch_candidates else None
            self.energy_col = energy_candidates[0] if energy_candidates else None

        pitch = np.nan
        energy = np.nan

        if self.pitch_col is not None and self.pitch_col in df.columns:
            pitch_series = pd.to_numeric(df[self.pitch_col], errors="coerce")
            pitch_series = pitch_series.replace([0, -np.inf, np.inf], np.nan)
            pitch = float(np.nanmedian(pitch_series.values))

        if self.energy_col is not None and self.energy_col in df.columns:
            energy_series = pd.to_numeric(df[self.energy_col], errors="coerce")
            energy = float(np.nanmedian(energy_series.values))

        return pitch, energy

    # -------------------------
    # finalize calibration
    # -------------------------
    def finalize_calibration(self):
        st = self.state

        if len(st.pitch_vals_cal) >= 3:
            st.mu_pitch = float(np.nanmean(st.pitch_vals_cal))
            st.sd_pitch = float(np.nanstd(st.pitch_vals_cal) + 1e-8)
        else:
            st.mu_pitch = 0.0
            st.sd_pitch = 1.0

        if len(st.energy_vals_cal) >= 3:
            st.mu_energy = float(np.nanmean(st.energy_vals_cal))
            st.sd_energy = float(np.nanstd(st.energy_vals_cal) + 1e-8)
        else:
            st.mu_energy = 0.0
            st.sd_energy = 1.0

        if len(st.pause_vals_cal) >= 1:
            mu_pause = float(np.mean(st.pause_vals_cal))
            sd_pause = float(np.std(st.pause_vals_cal) + 1e-8)
            st.tau_pause = mu_pause + K_PAUSE * sd_pause
        else:
            st.tau_pause = FALLBACK_TAU_PAUSE

        st.calibration_done = True

        print("\n[Calibration complete]")
        print(f"  pitch baseline: mu={st.mu_pitch:.3f}, sd={st.sd_pitch:.3f}")
        print(f"  energy baseline: mu={st.mu_energy:.3f}, sd={st.sd_energy:.3f}")
        print(f"  pause threshold: tau={st.tau_pause:.3f}s\n")

    # -------------------------
    # process one chunk
    # -------------------------
    def process_chunk(self, chunk: np.ndarray):
        st = self.state
        is_speech = self.is_speech_chunk(chunk)
        pitch, energy = self.extract_pitch_energy(chunk)

        now_start = st.processed_sec
        now_end = st.processed_sec + BLOCK_SEC
        st.processed_sec = now_end

        # ---------------------
        # calibration phase
        # ---------------------
        if not st.calibration_done:
            if is_speech:
                if not np.isnan(pitch):
                    st.pitch_vals_cal.append(pitch)
                if not np.isnan(energy):
                    st.energy_vals_cal.append(energy)
                st.current_silence_sec = 0.0
            else:
                st.current_silence_sec += BLOCK_SEC

            # calibration pause accumulation
            if is_speech and st.current_silence_sec >= MICRO_PAUSE_SEC:
                st.pause_vals_cal.append(st.current_silence_sec)
                st.current_silence_sec = 0.0

            if st.processed_sec >= CALIBRATION_SEC:
                self.finalize_calibration()
            return

        # ---------------------
        # evaluation phase
        # ---------------------
        st.eval_time_sec += BLOCK_SEC

        if is_speech:
            # end silence run
            if st.current_silence_sec > 0:
                if st.current_silence_sec > st.tau_pause:
                    st.eval_pause_excess_sec += (st.current_silence_sec - st.tau_pause)
                    self.popup.show("정적이 너무 길게 유지되고 있습니다", key="pause_long")
                st.current_silence_sec = 0.0

            # pitch
            if not np.isnan(pitch):
                z_pitch = (pitch - st.mu_pitch) / st.sd_pitch
                st.eval_pitch_total += 1

                if z_pitch > Z_HIGH:
                    self.popup.show("지금 피치가 너무 높습니다", key="pitch_high")
                    st.eval_pitch_viol += 1
                elif z_pitch < -Z_LOW:
                    self.popup.show("지금 피치가 너무 낮습니다", key="pitch_low")
                    st.eval_pitch_viol += 1

            # energy
            if not np.isnan(energy):
                z_energy = (energy - st.mu_energy) / st.sd_energy
                st.eval_energy_total += 1

                if z_energy > Z_HIGH:
                    self.popup.show("지금 목소리가 너무 큽니다", key="energy_high")
                    st.eval_energy_viol += 1
                elif z_energy < -Z_LOW:
                    self.popup.show("지금 목소리가 너무 작습니다", key="energy_low")
                    st.eval_energy_viol += 1

        else:
            st.current_silence_sec += BLOCK_SEC

    # -------------------------
    # compute summary scores
    # -------------------------
    def get_summary(self):
        st = self.state
        r_pitch = st.eval_pitch_viol / max(1, st.eval_pitch_total)
        r_energy = st.eval_energy_viol / max(1, st.eval_energy_total)
        r_pause = st.eval_pause_excess_sec / max(1e-6, st.eval_time_sec)

        S_pitch = 100.0 * np.exp(-3.0 * r_pitch)
        S_energy = 100.0 * np.exp(-3.0 * r_energy)
        S_pause = 100.0 * np.exp(-3.0 * r_pause)

        return {
            "r_pitch": r_pitch,
            "r_energy": r_energy,
            "r_pause": r_pause,
            "S_pitch": float(np.clip(S_pitch, 0, 100)),
            "S_energy": float(np.clip(S_energy, 0, 100)),
            "S_pause": float(np.clip(S_pause, 0, 100)),
        }

    # -------------------------
    # queue polling
    # -------------------------
    def poll_queue(self):
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            self.process_chunk(chunk)

        if not self.stop_event.is_set():
            self.popup.root.after(50, self.poll_queue)

    # -------------------------
    # start / stop
    # -------------------------
    def start(self):
        print("🎤 Realtime speech monitor started")
        print(f"Calibration: first {CALIBRATION_SEC:.1f}s")
        print("Close the window or press Ctrl+C to stop.\n")

        self.stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self.audio_callback,
        )
        self.stream.start()
        self.popup.root.after(50, self.poll_queue)

    def stop(self):
        self.stop_event.set()
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        summary = self.get_summary()
        print("\n===== FINAL SUMMARY =====")
        print(f"Pitch violation ratio : {summary['r_pitch']:.3f}")
        print(f"Energy violation ratio: {summary['r_energy']:.3f}")
        print(f"Pause excess ratio    : {summary['r_pause']:.3f}")
        print(f"Pitch Score           : {summary['S_pitch']:.1f}")
        print(f"Energy Score          : {summary['S_energy']:.1f}")
        print(f"Pause Score           : {summary['S_pause']:.1f}")
        print("=========================\n")


# ============================================================
# Main
# ============================================================
def main():
    root = tk.Tk()
    root.withdraw()  # 메인 루트는 숨기고 popup만 사용

    popup_manager = PopupManager(root)
    monitor = RealtimeSpeechMonitor(popup_manager)

    def on_close():
        monitor.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        monitor.start()
        root.mainloop()
    except KeyboardInterrupt:
        monitor.stop()
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()