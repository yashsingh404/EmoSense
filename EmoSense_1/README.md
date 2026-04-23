# EmoSense — Real-Time Emotion Detection with FER2013

A complete deep learning project that trains a CNN on the **FER2013** dataset and serves real-time emotion detection from your webcam via a beautiful web interface.

---

## Features

- **Deep CNN** with 4 convolutional blocks + batch normalization + dropout
- Trained on **FER2013** (35,887 facial expression images, 7 classes)
- **Data augmentation** (flip, rotate, shift, zoom) to reduce overfitting
- **Real-time webcam** detection via browser — face bounding box + live emotion label
- **Confidence bars** for all 7 emotion classes
- **Session stats** — FPS, emotion history, top emotion, average confidence
- **Snapshot** capture with overlaid detections
- **Browser demo mode** — works without a trained model for UI preview

## Emotion Classes

| Class | Color |
|-------|-------|
| 😠 Angry | Red |
| 🤢 Disgust | Green |
| 😨 Fear | Purple |
| 😊 Happy | Yellow |
| 😢 Sad | Blue |
| 😲 Surprise | Orange |
| 😐 Neutral | Gray |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FER2013

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract
3. Place `fer2013.csv` in the `data/` folder:

```
emotion_detection/
├── data/
│   └── fer2013.csv   ← here
```

### 3. Explore the dataset (optional)

```bash
python explore_data.py
```

Outputs plots to `model/plots/`:
- `samples.png` — sample image per class
- `class_distribution.png` — class balance per split

### 4. Train the model

```bash
python train_model.py
```

- Takes ~30–60 min on a GPU, several hours on CPU
- Saves the best model to `model/emotion_model.h5`
- Outputs training plots and confusion matrix to `model/plots/`

**Expected accuracy:** ~65–68% on FER2013 test set (state-of-the-art is ~72%)

### 5. Run the server

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### 6. Use the web UI

1. Click **Start Camera** — allow webcam access
2. Make sure **Server** mode is selected
3. Watch your emotions detected in real time!

> **No model yet?** Switch to **Browser (Demo)** mode to preview the UI with simulated predictions.

---

## Model Architecture

```
Input (48×48×1)
│
├─ Conv2D(64) + BN + Conv2D(64) + BN → MaxPool → Dropout(0.25)
├─ Conv2D(128) + BN + Conv2D(128) + BN → MaxPool → Dropout(0.25)
├─ Conv2D(256) + BN + Conv2D(256) + BN → MaxPool → Dropout(0.25)
├─ Conv2D(512) + BN → GlobalAvgPool
│
├─ Dense(512) + BN + Dropout(0.5)
├─ Dense(256) + Dropout(0.3)
└─ Dense(7, softmax)
```

**Optimizer:** Adam (lr=1e-3, ReduceLROnPlateau)  
**Loss:** Categorical cross-entropy  
**Regularization:** L2 (1e-4) + BatchNorm + Dropout

---

## Project Structure

```
emotion_detection/
├── train_model.py      ← Model training script
├── explore_data.py     ← Dataset visualization
├── app.py              ← Flask web server
├── requirements.txt    ← Python dependencies
├── README.md
├── data/
│   └── fer2013.csv     ← (you download this)
├── model/
│   ├── emotion_model.h5  ← saved after training
│   ├── logs/             ← TensorBoard logs
│   └── plots/            ← training curves, confusion matrix
└── templates/
    └── index.html        ← Web UI
```

---

## Improving Accuracy

- **Transfer learning**: Start from a pretrained ResNet/MobileNet backbone
- **Oversampling**: FER2013 is imbalanced (Disgust << Happy). Oversample minority classes.
- **Ensemble**: Average predictions from 2–3 models trained with different seeds
- **Label smoothing**: Adds small label noise to improve generalization
- **Test-time augmentation (TTA)**: Average predictions over horizontally flipped versions

---

## TensorBoard

```bash
tensorboard --logdir model/logs
```

Then open http://localhost:6006

---

## License

MIT — free to use, modify, and distribute.
