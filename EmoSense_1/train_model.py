"""
Emotion Detection Model Training - FER2013 Dataset
====================================================
CNN architecture trained on FER2013 dataset.

FER2013 Dataset Setup (image-folder format):
--------------------------------------------
Your archive.zip is already in the right format!

1. Extract archive.zip anywhere, e.g.:
       unzip archive.zip -d data/

   This produces:
       data/
         train/
           angry/    disgust/   fear/
           happy/    neutral/   sad/    surprise/
         test/
           angry/    disgust/   fear/
           happy/    neutral/   sad/    surprise/

2. Set TRAIN_DIR and TEST_DIR below to point to those folders.
3. Run: python train_model.py

No CSV needed — images are loaded directly from subfolders.
Each subfolder name IS the emotion label.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
# ↓↓ SET THESE TO WHERE YOU EXTRACTED archive.zip ↓↓
TRAIN_DIR = "data/train"   # folder with 7 subfolders (angry, happy, ...)
TEST_DIR  = "data/test"    # folder with 7 subfolders

# Emotion label order — must match subfolder names (case-insensitive handled below)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_DISPLAY = [e.capitalize() for e in EMOTIONS]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 60
VAL_SPLIT = 0.1          # 10% of training data used for validation
MODEL_SAVE_PATH = "model/emotion_model.h5"
PLOTS_DIR = "model/plots"

os.makedirs("model", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_images_from_folder(folder):
    """
    Load all 48x48 grayscale images from a folder structured as:
        folder/
          angry/   *.jpg
          happy/   *.jpg
          ...
    Returns numpy arrays X (N,48,48,1) float32 and y (N,) int.
    """
    import cv2

    X, y = [], []
    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(folder, emotion)
        if not os.path.isdir(emotion_dir):
            # Try capitalized version (e.g. "Angry" instead of "angry")
            emotion_dir = os.path.join(folder, emotion.capitalize())
        if not os.path.isdir(emotion_dir):
            print(f"  WARNING: folder not found: {emotion_dir} — skipping")
            continue

        files = [f for f in os.listdir(emotion_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {emotion:10s}: {len(files)} images")

        for fname in files:
            img_path = os.path.join(emotion_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
            X.append(img)
            y.append(emotion_to_idx[emotion])

    return np.array(X, dtype='float32'), np.array(y, dtype='int32')


def load_dataset():
    """Load train+val from TRAIN_DIR and test from TEST_DIR."""
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"TRAIN_DIR '{TRAIN_DIR}' not found.\n"
            "Extract archive.zip first:\n"
            "    unzip archive.zip -d data/\n"
            "Then set TRAIN_DIR = 'data/train' and TEST_DIR = 'data/test' above."
        )
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"TEST_DIR '{TEST_DIR}' not found.\n"
            "Check that archive.zip was extracted correctly."
        )

    print(f"\nLoading training images from: {TRAIN_DIR}")
    X_all, y_all = load_images_from_folder(TRAIN_DIR)

    print(f"\nLoading test images from: {TEST_DIR}")
    X_test, y_test = load_images_from_folder(TEST_DIR)

    # Split off a validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y_all
    )

    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val   = keras.utils.to_categorical(y_val,   NUM_CLASSES)
    y_test  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"\nDataset loaded:")
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)}   samples  ({int(VAL_SPLIT*100)}% of train)")
    print(f"  Test  : {len(X_test)}  samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── Augmentation ─────────────────────────────────────────────────────────────
def build_augmentor():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )


# ── Model Architecture ────────────────────────────────────────────────────────
def build_model():
    """
    Deep CNN with residual-style blocks for FER2013.
    Architecture inspired by VGG + modern regularization.
    """
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(512, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier head
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Callbacks ────────────────────────────────────────────────────────────────
def build_callbacks():
    return [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True,
                        monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=12, restore_best_weights=True,
                      monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6,
                          monitor='val_loss', verbose=1),
    ]


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0d0d1a')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#aaa')

    ax1.plot(history.history['accuracy'],   color='#7c6bff', lw=2, label='Train')
    ax1.plot(history.history['val_accuracy'], color='#00e5ff', lw=2, label='Val')
    ax1.set_title('Accuracy', color='white', fontsize=14)
    ax1.set_xlabel('Epoch', color='#aaa'); ax1.set_ylabel('Accuracy', color='#aaa')
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')

    ax2.plot(history.history['loss'],     color='#ff6b6b', lw=2, label='Train')
    ax2.plot(history.history['val_loss'], color='#ffd166', lw=2, label='Val')
    ax2.set_title('Loss', color='white', fontsize=14)
    ax2.set_xlabel('Epoch', color='#aaa'); ax2.set_ylabel('Loss', color='#aaa')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/training_history.png", dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a')
    print(f"Saved training plot to {PLOTS_DIR}/training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('True', color='white', fontsize=12)
    ax.set_title('Confusion Matrix — FER2013', color='white', fontsize=14)
    plt.xticks(color='#ccc', rotation=45); plt.yticks(color='#ccc', rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a')
    print(f"Saved confusion matrix to {PLOTS_DIR}/confusion_matrix.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        return

    model = build_model()
    model.summary()

    augmentor = build_augmentor()
    train_gen = augmentor.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

    print("\nStarting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=build_callbacks()
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS_DISPLAY))

    plot_training(history)
    plot_confusion_matrix(y_true, y_pred)

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print("Now run:  python app.py  to start the real-time detection server.")


if __name__ == "__main__":
    main()
