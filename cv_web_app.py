from flask import Flask, render_template, request
import numpy as np
import cv2
import base64
import sys
import json

app = Flask(__name__, template_folder="templates")

# Load the ONNX model
net = cv2.dnn.readNetFromONNX('model.onnx')

# Softmax function for converting raw outputs to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))  # numerical stability
    return e_x / np.sum(e_x)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    try:
        data_url = request.form.get('data')
        if not data_url or ',' not in data_url:
            raise ValueError("Invalid image data")

        # Decode base64 image
        encoded = data_url.split(',')[1]
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Failed to decode image")

        # Preprocess image
        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(28, 28), mean=(0,), swapRB=False)
        blob = blob.astype(np.float32)

        # Predict
        net.setInput(blob)
        out = net.forward()
        out = softmax(out.flatten())  # ✅ Apply softmax

        class_id = int(np.argmax(out))
        confidence = float(out[class_id])  # between 0 and 1

        print(f"[INFO] Prediction: {class_id}, Confidence: {confidence:.4f}", file=sys.stdout)

        return json.dumps({'success': True, 'class': class_id, 'confidence': confidence}), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return json.dumps({'success': False, 'error': str(e)}), 500, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
