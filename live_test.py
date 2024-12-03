import numpy as np
import librosa
import sounddevice as sd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Töltsük be a betanított modellt
model = load_model('drone_noise_detector_model.keras')

# Definiáljuk a mintavételi frekvenciát és az egyes szegmensek hosszát (1 másodperc)
sample_rate = 16000  # Eredeti mintavételi frekvencia valós idejű rögzítéshez
segment_duration = 1.0


scaler = StandardScaler()
scaler.mean_ = np.zeros(40)
scaler.scale_ = np.ones(40)


def extract_mfcc(audio_segment, sample_rate):

    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Normalizáljuk az MFCC-t [0, 1] tartományra
    normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
    return normalized_mfcc


def audio_callback(indata, frames, time, status):

    if status:
        print(status)

    # Gondoskodunk róla, hogy az audio mono legyen
    if indata.ndim > 1:
        indata = indata[:, 0]

    # Reszampeljük az audio-t 16000 Hz-re
    indata_resampled = librosa.resample(indata, orig_sr=sample_rate, target_sr=16000)

    # Kinyerjük az MFCC jellemzőket az átméretezett szegmensből
    mfcc_features = extract_mfcc(indata_resampled, 16000)  # Most már 16000 Hz mintavételi frekvencia

    # Átalakítjuk a bemenet formáját a modell számára
    mfcc_features = np.expand_dims(mfcc_features, axis=0)

    # Predikció készítése
    prediction = model.predict(mfcc_features)

    # A valószínűség kinyerése
    probability = prediction[0][0]

    # Küszöbérték beállítása, ha szükséges
    threshold = 0.5

    # A drónos osztály vagy a nem drónos észlelés valószínűségének kiírása
    if probability > threshold:
        print(f"Drone noise detected with {probability * 100:.2f}% confidence")
    else:
        print(f"No drone noise detected with {(1 - probability) * 100:.2f}% confidence")

# Az audio stream indítása
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate,
                    blocksize=int(sample_rate * segment_duration)):
    print("Listening for drone noise... Press Ctrl+C to stop.")
    sd.sleep(int(segment_duration * 1000) * 60)  # Listen for 60 seconds

print("Stopped listening.")
