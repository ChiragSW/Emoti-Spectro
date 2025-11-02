import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0

train_dir = "spectros_rgb_split/train"
val_dir   = "spectros_rgb_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20
INPUT_SHAPE = (224, 224, 3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

def build_cnn_bigru(input_shape, num_classes):
    # EfficientNetB0 base
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)


    for layer in base_model.layers[-40:]:  # unfreeze last 40 layers
        layer.trainable = True

    x = base_model.output  # shape (7, 7, 1280)
    x = layers.Reshape((x.shape[1], -1))(x)  # (time_steps=7, features=7*1280)


    x = layers.Bidirectional(layers.GRU(256, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


model = build_cnn_bigru(INPUT_SHAPE, NUM_CLASSES)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_logs", exist_ok=True)

callbacks = [
    ModelCheckpoint("checkpoints/cnn_bigru_best.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    CSVLogger("training_logs/cnn_bigru_log.csv", append=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2,
    callbacks=callbacks
)

model.save("cnn_bigru_finetuned_final.keras")
print("Training complete")
