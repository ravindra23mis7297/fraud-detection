"""
train_image_model.py
====================
Instructions:
    1. Organize your dataset as:
           dataset/
               normal/    <- images of untampered food
               tampered/  <- images of tampered food
    2. Install dependencies:
           pip install tensorflow pillow numpy
    3. Run:
           python train_image_model.py
    4. Output: image_fraud_model.h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"          # Root folder containing normal/ and tampered/
IMG_SIZE      = (128, 128)         # Resize all images to this resolution
BATCH_SIZE    = 32
EPOCHS        = 30
VALIDATION_SPLIT = 0.2
MODEL_OUTPUT  = "image_fraud_model.h5"
SEED          = 42

# ── Data Generators ───────────────────────────────────────────────────────────
# Augment training data to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",          # 0 = normal, 1 = tampered
    subset="training",
    seed=SEED,
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED,
)

print(f"Class indices: {train_generator.class_indices}")   # Confirm label mapping
print(f"Training samples  : {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ── Model Architecture ────────────────────────────────────────────────────────
# Lightweight CNN suited for binary image classification
def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # Probability of tampering
    ])
    return model

model = build_model()
model.summary()

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUTPUT, monitor="val_auc", save_best_only=True,
                    mode="max", verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)

print(f"\nModel saved to: {MODEL_OUTPUT}")
print(f"Best validation AUC: {max(history.history['val_auc']):.4f}")
