import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===================================================
# CONFIG
# ===================================================
MODEL_PATH = "checkpoints/cnn_gru_best.keras"             
DATA_DIR = "spectros_rgb_split"                   # same split folder used in training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
COLOR_MODE = "rgb"                         
CSV_SAVE_PATH = "results_cnn_gru.csv"

# ===================================================
# LOAD MODEL
# ===================================================
print("[INFO] Loading model...")
model = tf.keras.models.load_model("MODEL_PATH", compile=False, safe_mode=False)

# ===================================================
# DATA GENERATOR (Validation)
# ===================================================
datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===================================================
# PREDICTIONS
# ===================================================
print("[INFO] Predicting...")
pred_probs = model.predict(val_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# ===================================================
# METRICS
# ===================================================
print("[INFO] Computing metrics...")
report_dict = classification_report(
    true_classes,
    pred_classes,
    target_names=class_labels,
    output_dict=True,
    digits=4
)

accuracy = accuracy_score(true_classes, pred_classes)
print(f"\nOverall Accuracy: {accuracy:.4f}\n")

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()
report_df["accuracy"] = accuracy

# Save to CSV
os.makedirs("metrics", exist_ok=True)
report_df.to_csv(os.path.join("metrics", CSV_SAVE_PATH), index=True)
print(f"[INFO] Metrics saved to metrics/{CSV_SAVE_PATH}")
