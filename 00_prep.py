import os
import shutil
import random
from tqdm import tqdm

SRC_DIR = "spectros_rgb"
DEST_DIR = "spectros_rgb_split"
train_ratio = 0.8  # 80% train 20% val

os.makedirs(os.path.join(DEST_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val"), exist_ok=True)

for emotion in os.listdir(SRC_DIR):
    emotion_folder = os.path.join(SRC_DIR, emotion)
    if not os.path.isdir(emotion_folder):
        continue

    files = os.listdir(emotion_folder)
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    os.makedirs(os.path.join(DEST_DIR, "train", emotion), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "val", emotion), exist_ok=True)

    for f in tqdm(train_files, desc=f"Train {emotion}"):
        shutil.copy(os.path.join(emotion_folder, f), os.path.join(DEST_DIR, "train", emotion, f))

    for f in tqdm(val_files, desc=f"Val {emotion}"):
        shutil.copy(os.path.join(emotion_folder, f), os.path.join(DEST_DIR, "val", emotion, f))

# sanity check
from collections import Counter
import os

for split in ["train", "val"]:
    counts = {emo: len(os.listdir(os.path.join("spectros_rgb_split", split, emo))) 
              for emo in os.listdir(os.path.join("spectros_rgb_split", split))}
    print(split, counts)
