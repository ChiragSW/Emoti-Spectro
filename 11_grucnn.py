import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

train_dir = "spectros_rgb_split/train"
val_dir   = "spectros_rgb_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 50
COLOR_MODE = 'rgb'
INPUT_SHAPE = (224, 224, 3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# for newer tf versions
def build_cnn_gru(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # CNN feature extractor
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # compute shape
    shape = tf.keras.backend.int_shape(x)
    time_steps = shape[1] or 1
    feature_dim = shape[2] * shape[3]
    x = layers.Reshape((time_steps, feature_dim))(x)

    # GRU
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="CNN_GRU_Model")
    return model


model = build_cnn_gru(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_logs", exist_ok=True)

callbacks = [
    ModelCheckpoint("checkpoints/cnn_gru_best.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    CSVLogger("training_logs/cnn_gru_log.csv", append=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("cnn_gru_final.keras", save_format="keras_v3")
print("Training complete. Model saved as cnn_gru_final.keras")
