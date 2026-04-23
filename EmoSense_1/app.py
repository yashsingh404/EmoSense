"""
Real-Time Emotion Detection Server
====================================
Serves the web UI and exposes a REST endpoint to run inference
on frames sent from the browser webcam.

Run: python app.py
Then open: http://localhost:5000
"""

import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Lazy imports so the server boots even without TF installed yet
_model = None
_face_cascade = None

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😊',
    'Sad':      '😢',
    'Surprise': '😲',
    'Neutral':  '😐',
}
EMOTION_COLORS = {
    'Angry':    '#ff4d4d',
    'Disgust':  '#7cb518',
    'Fear':     '#a855f7',
    'Happy':    '#ffd166',
    'Sad':      '#4a9eff',
    'Surprise': '#ff9f1c',
    'Neutral':  '#94a3b8',
}
MODEL_PATH = "model/emotion_model.h5"
IMG_SIZE = 48

app = Flask(__name__)
CORS(app)


def get_model():
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
            _model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded.")
        except ImportError:
            raise RuntimeError("TensorFlow not installed. Run: pip install tensorflow")
    return _model


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError("Could not load Haar cascade.")
    return _face_cascade


def decode_image(data_url):
    """Decode a base64 data URL to a numpy array (BGR)."""
    import cv2
    header, b64 = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def detect_faces_and_predict(frame_bgr):
    """Detect faces, run emotion prediction, return list of detections."""
    import cv2

    model = get_model()
    cascade = get_face_cascade()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detections = []
    for (x, y, w, h) in faces:
        # Crop and preprocess
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_norm = roi_resized.astype('float32') / 255.0
        roi_input = roi_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        preds = model.predict(roi_input, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        emotion = EMOTIONS[top_idx]
        confidence = float(preds[top_idx])

        all_scores = {EMOTIONS[i]: round(float(preds[i]) * 100, 1) for i in range(len(EMOTIONS))}
        sorted_scores = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)

        detections.append({
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
            'emotion': emotion,
            'emoji': EMOTION_EMOJIS[emotion],
            'color': EMOTION_COLORS[emotion],
            'confidence': round(confidence * 100, 1),
            'scores': sorted_scores,
        })

    return detections


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           emotions=EMOTIONS,
                           emotion_colors=EMOTION_COLORS,
                           emotion_emojis=EMOTION_EMOJIS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        frame = decode_image(data['image'])
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        detections = detect_faces_and_predict(frame)
        return jsonify({'detections': detections, 'face_count': len(detections)})

    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'model_missing': True}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({'status': 'ok', 'model_ready': model_ready})


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Emotion Detection Server")
    print("="*55)
    if not os.path.exists(MODEL_PATH):
        print(f"  WARNING: Model not found at {MODEL_PATH}")
        print("  Run train_model.py first to train the model.")
    else:
        print(f"  Model found: {MODEL_PATH}")
    print("  Starting on http://localhost:5000")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
