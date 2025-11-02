# fine tuned

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

train_dir = "spectros_rgb_split/train"
val_dir = "spectros_rgb_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 50
EPOCHS_FREEZE = 10
EPOCHS_FINE_TUNE = 20

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

datagen_val = ImageDataGenerator(rescale=1./255)

train_gen = datagen_train.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = datagen_val.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_logs", exist_ok=True)

callbacks = [
    ModelCheckpoint("checkpoints/mobilenetv2_best_fine.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    CSVLogger("training_logs/mobilenetv2_fine_log.csv", append=True)
]

print("\nTraining top layers only...")
history_freeze = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FREEZE,
    callbacks=callbacks
)

print("\nFine-tuning last 40 layers...")

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks
)

model.save("mobilenetv2_emotion_fine.keras")
print("\n Training complete. Model saved as mobilenetv2_emotion_fine.keras")