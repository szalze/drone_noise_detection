import numpy as np
import librosa
from tensorflow.keras.models import load_model


def extract_mfcc(audio_segment, sample_rate):
    """
    Extract MFCC from the audio segment and normalize it
    """
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Normalize MFCC to range [0, 1]
    normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
    return normalized_mfcc


# Load the trained model
model = load_model('best_drone_noise_detector_model.keras')

# Define the path to the new audio file you want to test
test_audio_path = r'C:\Drone_Dataset\teszt\drone\GX015000.wav'

# Load the entire audio file
audio, sample_rate = librosa.load(test_audio_path, res_type="kaiser_fast", mono=True)

# Resample the audio to 16000 Hz
audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

# Define the duration of each segment (1 second)
segment_duration = 1.0
segment_samples = int(segment_duration * 16000)  # 16000 Hz sampling rate

# Initialize counters for drone noise detection
drone_detected_count = 0
total_segments = 0

# Process the audio file in 1-second segments
for start in range(0, len(audio_resampled), segment_samples):
    end = start + segment_samples
    if end > len(audio_resampled):
        break
    audio_segment = audio_resampled[start:end]

    # Extract MFCC features from the segment
    mfcc_features = extract_mfcc(audio_segment, 16000)  # 16000 Hz sample rate now
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    mfcc_features = np.expand_dims(mfcc_features, axis=2)

    # Make a prediction
    prediction = model.predict(mfcc_features)

    # Get the probability of drone noise detection
    probability = prediction[0][0]

    # Check if the classes are swapped
    # Assuming the model outputs 1 for drone and 0 for no drone
    # If the classes are swapped, we need to invert the probability
    if probability > 0.5:
        print(f"Segment {total_segments + 1}: {probability * 100:.2f}% chance of drone noise (Drone detected).")
        drone_detected_count += 1
    else:
        print(f"Segment {total_segments + 1}: {probability * 100:.2f}% chance of drone noise (No drone detected).")

    total_segments += 1

# Calculate the percentage of segments that detected drone noise
drone_detection_percentage = (drone_detected_count / total_segments) * 100

# Print the result
print(f"Drone noise detected in {drone_detection_percentage:.2f}% of the segments.")