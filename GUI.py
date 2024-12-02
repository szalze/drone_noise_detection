import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk

class DroneNoiseDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Noise Detector")
        self.is_recording = False
        self.audio_buffer = []
        self.sample_rate = 16000
        self.segment_duration = 1.0  # Szegmens időtartama másodpercben
        self.no_signal_detected = False

        # Modell betöltése
        self.model = load_model('best_drone_noise_detector_model.keras')

        # GUI elrendezés
        self.label = ttk.Label(root, text="Drone Noise Detector", font=("Arial", 16))
        self.label.pack(pady=10)

        # Drón jelző
        self.drone_indicator = tk.Label(root, text="DRONE", bg="gray", fg="white", font=("Arial", 16), width=10)
        self.drone_indicator.pack(pady=10)

        # Felvétel gomb
        self.start_button = tk.Button(root, text="Recording", bg="lightgray", font=("Arial", 12),
                                       command=self.start_recording)
        self.start_button.pack(pady=5)

        # Stop gomb
        self.stop_button = tk.Button(root, text="Stop", bg="lightgray", font=("Arial", 12),
                                      command=self.stop_recording)
        self.stop_button.pack(pady=5)

        # Nincs jel jelző
        self.no_signal_label = tk.Label(root, text="", font=("Arial", 10), fg="red")
        self.no_signal_label.pack(pady=10)

    def extract_mfcc(self, audio_segment):
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=16000, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        # Normalizálás 0-1 közé
        normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
        return normalized_mfcc

    def audio_callback(self, indata, frames, time, status):

        if status:
            print(status)

        # Mono audio biztosítása
        if indata.ndim > 1:
            indata = indata[:, 0]

        # Ellenőrizzük hogy halk-e a jel
        if np.abs(indata).max() < 0.01:  # Küszöb csend érzékeléséhez
            self.no_signal_label.config(text="Nincs bemeneti jel!")
            self.no_signal_detected = True
        else:
            self.no_signal_label.config(text="")
            self.no_signal_detected = False

        self.audio_buffer.extend(indata)

        # Hangfeldolgozás, ha elegendő minta összegyűlt
        samples_needed = int(self.sample_rate * self.segment_duration)
        if len(self.audio_buffer) >= samples_needed:
            segment = np.array(self.audio_buffer[:samples_needed])
            self.audio_buffer = self.audio_buffer[samples_needed:]

            # Átmintavételezés 16 kHz-re
            segment_resampled = librosa.resample(segment, orig_sr=self.sample_rate, target_sr=16000)

            # Jellemzők kinyerése
            mfcc_features = self.extract_mfcc(segment_resampled)
            mfcc_features = np.expand_dims(mfcc_features, axis=0)
            mfcc_features = np.expand_dims(mfcc_features, axis=2)

            # Predikció
            prediction = self.model.predict(mfcc_features)
            probability = prediction[0][0]
            threshold = 0.5

            # Drón indikátor frissítése (ha nincs jel, nem változik)
            if not self.no_signal_detected:
                if probability > threshold:
                    self.drone_indicator.config(bg="red")
                else:
                    self.drone_indicator.config(bg="gray")
    # Felvétel indítása
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer = []
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1,
                                         samplerate=self.sample_rate,
                                         blocksize=int(self.sample_rate * self.segment_duration))
            self.stream.start()
            self.flash_start_button()

    # Felvétel leállítása
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.start_button.config(bg="lightgray")  # Alapértelmezett szín visszaállítása

    def flash_start_button(self):
        if self.is_recording:
            current_color = self.start_button.cget("bg")
            new_color = "red" if current_color == "lightgray" else "lightgray"
            self.start_button.config(bg=new_color)
            self.root.after(500, self.flash_start_button)  # Villogás 500 ms-enként

# Alkalmazás inicializálása
if __name__ == "__main__":
    root = tk.Tk()
    app = DroneNoiseDetectorApp(root)
    root.mainloop()
