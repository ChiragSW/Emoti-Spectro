import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
# Paths
RAW_DATASET = "raw_data"
PROCESSED_DATASET = "fin_data"
os.makedirs(PROCESSED_DATASET, exist_ok=True)

# Parameters
TARGET_SR = 22050 #sampling rate
TARGET_DURATION = 3  #seconds
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

for emotion in os.listdir(RAW_DATASET):
    emotion_path = os.path.join(RAW_DATASET, emotion)
    save_emotion_path = os.path.join(PROCESSED_DATASET, emotion)
    os.makedirs(save_emotion_path, exist_ok=True)
    
    for file in tqdm(os.listdir(emotion_path), desc=f"Processing {emotion}"):
        file_path = os.path.join(emotion_path, file)
        
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=TARGET_SR)
            
            # Trim leading silence
            y, _ = librosa.effects.trim(y)
            
            # cut to 3 seconds
            if len(y) > TARGET_SAMPLES:
                y = y[:TARGET_SAMPLES]
            else:
                padding = TARGET_SAMPLES - len(y)
                y = np.pad(y, (0, padding))
            
            # Save normalized version
            save_path = os.path.join(save_emotion_path, file.replace('.mp3', '.wav').replace('.ogg', '.wav'))
            sf.write(save_path, y, TARGET_SR)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
