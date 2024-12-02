import os
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint


# MFCC jellemzők kinyerése a wav fájlokból
def extract_mfcc(audio_path, sr=16000):
    # Audiofájl betöltése
    audio, _ = librosa.load(audio_path, sr=sr, res_type="kaiser_fast", mono=True)

    # MFCC jellemzők kinyerése
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Normalizálás 0 és 1 közé
    normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
    return normalized_mfcc


# Augmentáció
def augment_audio(audio, sr):
    augmented_audios = []

    # Hangmagasság módosítása véletlenszerűen -2 és 2 félhang között
    pitch_shift = np.random.uniform(-2, 2)
    audio_pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
    augmented_audios.append(audio_pitch_shifted)

    # Időnyújtás 0.8 és 1.2 közötti aránnyal
    stretch_rate = np.random.uniform(0.8, 1.2)
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)

    # Az eredeti hossz megtartása: vágás vagy nullákkal kitöltés
    if len(audio_stretched) > len(audio):
        audio_stretched = audio_stretched[:len(audio)]
    else:
        audio_stretched = np.pad(audio_stretched, (0, max(0, len(audio) - len(audio_stretched))), "constant")

    augmented_audios.append(audio_stretched)

    return augmented_audios


# Adatok helye és osztályok definiálása
data_dir = r'C:\Drone_Dataset'
classes = ['non_drone', 'drone']

# Adatok összegyűjtése és címkézése
file_paths = []
labels = []

# Osztályokon belüli fájlok bejárása
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(class_dir, filename)
            file_paths.append(file_path)
            labels.append(class_name)

# DataFrame létrehozása
labels_df = pd.DataFrame({'file_path': file_paths, 'class': labels})
drone_files = labels_df[labels_df['class'] == 'drone']
non_drone_files = labels_df[labels_df['class'] == 'non_drone']

# Az osztályok kiegyensúlyozása
augment_count = int((len(non_drone_files) - len(drone_files)) / 2)
augmented_features = []
sr = 16000

if augment_count > 0:
    for i in range(augment_count):
        idx = i % len(drone_files)
        audio, _ = librosa.load(drone_files.iloc[idx]['file_path'], sr=sr, res_type="kaiser_fast", mono=True)
        augmentations = augment_audio(audio, sr)

        for augmented_audio in augmentations:
            mfcc_aug = librosa.feature.mfcc(y=augmented_audio, sr=sr, n_mfcc=40)
            mfcc_aug = np.mean(mfcc_aug.T, axis=0)
            augmented_features.append([mfcc_aug, 'drone'])

# Augmentált adatok hozzáadása a DataFrame-hez
augmented_df = pd.DataFrame(augmented_features, columns=['feature', 'class_label'])
print(f"Augmentáció kész: {len(augmented_df)} új minta készült.")

# Eredeti és augmentált jellemzők kombinálása
features = []

for index_num, row in labels_df.iterrows():
    # Fájlútvonal és osztály lekérése
    file_path = row["file_path"]
    class_label = row["class"]

    # Jellemzők kinyerése
    extract = extract_mfcc(file_path)

    # Kinyert jellemzők mentése a listába
    features.append([extract, class_label])

# Augmentált minták hozzáadása
features.extend(augmented_features)

# DataFrame létrehozása
features_df = pd.DataFrame(features, columns=['feature', 'class_label'])

# Jellemzők mentése CSV fájlba
features_df.to_csv('extracted_features.csv', index=False)
print("Tanító adatok elmentve.")

# Adatok előkészítése a tanításhoz
X = np.array(features_df["feature"].tolist())
y = np.array(features_df["class_label"].tolist())

# Címkék kódolása (0: non_drone, 1: drone)
label_encoder = LabelEncoder()
label_encoder.fit(['non_drone', 'drone'])
y = label_encoder.transform(y)

# Tanító és tesztadatok szétválasztása
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Validációs adatok elkülönítése
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Adatok átalakítása a bemenethez
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_validation = np.expand_dims(X_validation, axis=2)

# A CNN modell definiálása
model = Sequential()
input_shape = (40, 1)

# Első konvolúciós blokk
model.add(Conv1D(256, kernel_size=3, padding="same", strides=1, input_shape=input_shape, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=1, padding="same"))

# Második konvolúciós blokk
model.add(Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=1, padding="same"))

# Harmadik konvolúciós blokk
model.add(Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=1, padding="same"))

# Dropout réteg
model.add(Dropout(0.3))

# Flatten réteg
model.add(Flatten())

# Dense réteg
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Kimenet
model.add(Dense(1, activation="sigmoid"))

# Modell fordítása
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbackek definiálása
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=5, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_drone_noise_detector_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Modell tanítása
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_validation, y_validation), callbacks=[learning_rate_reduction, early_stopping, model_checkpoint])

# Modell mentése
model.save('drone_noise_detector_model.keras')
print("A modell elmentve.")

# Eredmények ábrázolása
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Tanulási pontosság')
plt.plot(history.history['val_accuracy'], label='Validációs pontosság')
plt.title('Modell pontosság')
plt.xlabel('Tanulási ciklus')
plt.ylabel('Pontosság')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Tanulási veszteség')
plt.plot(history.history['val_loss'], label='Validációs veszteség')
plt.title('Modell veszteség')
plt.xlabel('Tanulási ciklus')
plt.ylabel('Veszteség')
plt.legend()

plt.tight_layout()
plt.show()
