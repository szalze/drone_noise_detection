import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# MFCC jellemzők kinyerése klaszterezéshez
def extract_mfcc(file_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Összegyűjtjük az MFCC jellemzőket az összes WAV fájlhoz
mfcc_features = []
file_paths = []
output_dir = r''

for filename in os.listdir(output_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(output_dir, filename)
        mfcc_features.append(extract_mfcc(file_path))
        file_paths.append(file_path)

# MFCC jellemzők átalakítása numpy tömbbé a klaszterezéshez
mfcc_features = np.array(mfcc_features)

# PCA alkalmazása a dimenziók csökkentésére (vizualizációhoz)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(mfcc_features)

# K-Means klaszterezés
n_clusters = 0
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(mfcc_features)

# Klaszterek vizualizálása
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.xlabel('PCA komponens 1')
plt.ylabel('PCA komponens 2')
plt.colorbar(label='Klaszter')
plt.show()

# Egy reprezentatív fájl kiválasztása klaszterenként
cluster_representatives = {}
for i, label in enumerate(labels):
    if label not in cluster_representatives:
        cluster_representatives[label] = file_paths[i]

# A klaszterezett fájlok mentésére használt mappa
representative_dir = r''
os.makedirs(representative_dir, exist_ok=True)

# Számláló a fájlok sorszámozásához
file_counter = 1  # A számozás kezdőértéke

# Iterálás a klaszterek reprezentatív fájljain
for cluster, file_path in cluster_representatives.items():
    # Az eredeti fájlnév
    original_filename = os.path.basename(file_path)

    # Új fájlnév generálása (eredeti név megtartása)
    new_filename = f'{file_counter}_{original_filename}'

    # Az új fájl útvonalának meghatározása
    output_path = os.path.join(representative_dir, new_filename)

    # A fájl áthelyezése és átnevezése
    os.rename(file_path, output_path)

    # Növeljük a számlálót
    file_counter += 1
'''
# Az összes fájl törlése az eredeti output könyvtárból
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
'''
print(f'{len(cluster_representatives)} reprezentatív fájl lett elmentve ide: {representative_dir}. '
      f'Az összes feldolgozott fájl törölve lett innen: {output_dir}.')
