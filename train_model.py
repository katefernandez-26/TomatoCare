"""
Train a simple CNN for TomatoCare.

Expected folder structure:
dataset/
  Tomato___Bacterial_spot/
    image1.jpg
  Tomato___Early_blight/
    image2.jpg
  Tomato___healthy/
    image3.jpg
  ...

Use a reputable, correctly labeled tomato-leaf dataset. The model only recognizes
classes included in the training folders. Training can take a while and requires
TensorFlow installed on a compatible Python environment.
"""
import json
from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 12

if not DATA_DIR.exists():
    raise SystemExit("Create the dataset/ folder and add one subfolder per class.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
if len(class_names) < 2:
    raise SystemExit("Add images for at least two classes to train a classifier.")

with (BASE_DIR / "class_names.json").open("w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    augmentation,
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BASE_DIR / "tomato_disease_model.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

model.summary()
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Ensure the final model file exists even if no checkpoint was written.
model.save(BASE_DIR / "tomato_disease_model.keras")
print("Training complete.")
print("Saved tomato_disease_model.keras and class_names.json")
print("Classes:", class_names)
