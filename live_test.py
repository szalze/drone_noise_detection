import numpy as np
import librosa
import sounddevice as sd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Töltsük be a betanított modellt
model = load_model('drone_noise_detector_model.keras')

# Definiáljuk a mintavételi frekvenciát és az egyes szegmensek hosszát (1 másodperc)
sample_rate = 48000
segment_duration = 1.0

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
    mfcc_features = extract_mfcc(indata_resampled, 16000)

    # Átalakítjuk a bemenet formáját a modell számára
    mfcc_features = np.expand_dims(mfcc_features, axis=0)

    # Predikció készítése
    prediction = model.predict(mfcc_features)

    # Valószínűség kiszámítása
    probability = prediction[0][0]
    print(f'{probability * 100:.2f}% valószínűséggel drónzaj.')

# Audio stream indítása
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate,
                    blocksize=int(sample_rate * segment_duration)):
    print('Drónzaj észlelés... Nyomd meg a Ctrl+C-t a leállításhoz.')
    sd.sleep(int(segment_duration * 1000) * 60)

print('Leállt a felvétel.')
