import os
import librosa
import soundfile as sf

# Beállítások
input_dir = r'C:\Users\szalz\Downloads\Pro.Sound.Effects.Library.Chicago.Ambisonics.WAV\CHI Ambisonics Stereo'  # Az eredeti wav fájlok mappája.
output_dir = r'C:\Drone_Dataset\resampled_no_drone'  # A feldarabolt fájlok mentési mappája.
os.makedirs(output_dir, exist_ok=True)  # Ha a mappa nem létezik, hozzuk létre.

# Egyedi alapnév az output fájlokhoz (ha üres, az eredeti fájlnevet használja)
custom_base_name = ""

# Szelet hossza másodpercben
slice_duration = 1

# Célszámítási frekvencia újramintavételezéshez
target_sample_rate = 16000

# Fájlszámláló az egyedi számozáshoz
file_counter = 1

# Az összes .wav fájl feldolgozása a bemeneti mappában
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):  # Csak a .wav fájlokat kezeljük
        input_file = os.path.join(input_dir, filename)

        # Audió betöltése a célszámítási frekvenciával
        audio_data, sample_rate = librosa.load(input_file, sr=target_sample_rate, mono=False)

        # Ellenőrizzük, hogy mono vagy sztereó
        if len(audio_data.shape) == 1:
            # Mono eset
            channels = [audio_data]
        else:
            # Sztereó fájl esetén minden csatornát külön kezelünk
            channels = [audio_data[0], audio_data[1]]

        # Ha nincs egyedi név megadva, használjuk az eredeti fájlnevet
        base_name = custom_base_name if custom_base_name else os.path.splitext(filename)[0]

        # Minták mentése mindkét csatornához
        for channel_index, channel_data in enumerate(channels):
            channel_name = 'left' if channel_index == 0 else 'right'  # Csatorna megnevezése
            samples_per_slice = int(slice_duration * target_sample_rate)  # Szelet mérete mintákban
            total_samples = len(channel_data)  # Összes minta száma

            for i in range(0, total_samples, samples_per_slice):
                segment = channel_data[i:i + samples_per_slice]  # Szelet kivágása
                segment_duration = len(segment) / target_sample_rate  # Szelet időtartama másodpercben

                # Csak azokat a szeleteket mentjük, amik pontosan 1 másodpercesek
                if segment_duration == slice_duration:
                    output_file = os.path.join(
                        output_dir,
                        f"{base_name}_{file_counter}.wav"
                    )
                    sf.write(output_file, segment, target_sample_rate)  # Szelet mentése fájlba
                    print(f'Minta mentve: {output_file}')
                    file_counter += 1
