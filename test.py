import numpy as np
import librosa
from tensorflow.keras.models import load_model


def extract_mfcc(audio_segment, sample_rate):

    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Az MFCC normalizálása 0 és 1 közötti tartományra
    normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
    return normalized_mfcc


# A betanított modell betöltése
model = load_model('best_drone_noise_detector_model.keras')

# Az új tesztelendő audiofájl elérési útjának megadása
test_audio_path = r'C:\Drone_Dataset\teszt\drone\GX015000.wav'

# Az egész audiofájl betöltése
audio, sample_rate = librosa.load(test_audio_path, res_type="kaiser_fast", mono=True)

# Az audio újramintavételezése 16 kHz-re
audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

# Az szegmensek hosszának meghatározása (1 másodperc)
segment_duration = 1.0
segment_samples = int(segment_duration * 16000)  # 16000 Hz-es mintavételi ráta

# Számlálók inicializálása a drónzaj észleléséhez
drone_detected_count = 0
total_segments = 0

# Az audiofájl feldolgozása 1 másodperces szegmensekre bontva
for start in range(0, len(audio_resampled), segment_samples):
    end = start + segment_samples
    if end > len(audio_resampled):
        break
    audio_segment = audio_resampled[start:end]

    # MFCC jellemzők kinyerése a szegmensből
    mfcc_features = extract_mfcc(audio_segment, 16000)  # Most már 16000 Hz a mintavételi ráta
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    mfcc_features = np.expand_dims(mfcc_features, axis=2)

    # Predikció készítése
    prediction = model.predict(mfcc_features)

    # A drónzaj észlelésének valószínűségének kinyerése
    probability = prediction[0][0]

    # Csak a valószínűségi értéket gyűjtsük és számoljuk a találatokat
    if probability > 0.5:
        drone_detected_count += 1

    total_segments += 1

# Annak kiszámítása, hogy a szegmensek hány százalékában észleltek drónzajt
drone_detection_percentage = (drone_detected_count / total_segments) * 100

# Az eredmény kiírása
print(f"Drónzaj észlelve a szegmensek {drone_detection_percentage:.2f}%-ában.")

