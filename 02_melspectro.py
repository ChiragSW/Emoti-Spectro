import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

AUDIO_PATH = "fin_data"
SPEC_PATH = "spectros_gs"
os.makedirs(SPEC_PATH, exist_ok=True)

# imp for pretrained cnn
IMG_SIZE = (224, 224)
SAMPLE_RATE = 22050

for emotion in os.listdir(AUDIO_PATH):
    emotion_folder = os.path.join(AUDIO_PATH, emotion)
    save_folder = os.path.join(SPEC_PATH, emotion)
    os.makedirs(save_folder, exist_ok=True)

    for file in tqdm(os.listdir(emotion_folder), desc=f"Generating {emotion}"):
        try:
            file_path = os.path.join(emotion_folder, file)
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Plot and save as image
            plt.figure(figsize=(2.24, 2.24))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None, cmap='gray')
            plt.axis('off')
            
            save_path = os.path.join(save_folder, file.replace('.wav', '.png'))
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
