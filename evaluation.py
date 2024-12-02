import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_score, recall_score,accuracy_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from tensorflow.keras.models import load_model


# 1. Mappa bejárása és fájlok összegyűjtése
def load_files_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".wav")]


# 2. MFCC jellemzők kinyerése és normalizálása
def extract_normalized_mfcc_from_wav(file_path, segment_duration=1.0, sample_rate=16100, n_mfcc=40):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_duration * sr)
    mfcc_features = []

    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        if end > len(audio):
            break
        segment = audio[start:end]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
        mfcc_features.append(normalized_mfcc)

    return np.array(mfcc_features)


# 3. Mappák betöltése és feldolgozása
def process_wav_directories_with_normalization(drone_dir, non_drone_dir, segment_duration=1.0):
    drone_files = load_files_from_directory(drone_dir)
    non_drone_files = load_files_from_directory(non_drone_dir)

    drone_features = []
    non_drone_features = []

    for file in drone_files:
        drone_features.extend(extract_normalized_mfcc_from_wav(file, segment_duration))
    for file in non_drone_files:
        non_drone_features.extend(extract_normalized_mfcc_from_wav(file, segment_duration))

    X = np.array(drone_features + non_drone_features)
    y = np.array(["drone"] * len(drone_features) + ["non_drone"] * len(non_drone_features))
    return X, y


# 4. Mappák megadása
drone_directory = r"C:\Drone_Dataset\test\drone"
non_drone_directory = r"C:\Drone_Dataset\test\non_drone"

# 5. Adatok feldolgozása
X, y = process_wav_directories_with_normalization(drone_directory, non_drone_directory)

# 6. Modell betöltése
model = load_model("best_drone_noise_detector_model.keras")

# 7. Előrejelzés
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype(int)

# 8. Címkék átalakítása
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 9. Teljesítménymutatók számítása
accuracy = accuracy_score(y_encoded, y_pred)
precision = precision_score(y_encoded, y_pred)
recall = recall_score(y_encoded, y_pred)
f1 = f1_score(y_encoded, y_pred)
auc_score = roc_auc_score(y_encoded, y_pred_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc_score:.2f}")

# 10. Ábrák készítése
# 10.1. Konfúziós mátrix
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Osztályozó mátrix")
plt.show()

# 10.2. ROC görbe és AUC
fpr, tpr, _ = roc_curve(y_encoded, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], "k--", label="Véletlen találat (baseline)")
plt.title("ROC-görbe")
plt.xlabel("Hamis pozitív arány (FPR)")
plt.ylabel("Igaz pozitív arány (TPR)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 10.3. Precízió-recall görbe
precisions, recalls, thresholds = precision_recall_curve(y_encoded, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f"F1-score = {f1:.2f}")
plt.title("Precízió-visszahívás görbe")
plt.xlabel("Visszahívás")
plt.ylabel("Precízió")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
