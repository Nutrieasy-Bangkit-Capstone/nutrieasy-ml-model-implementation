from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as utils
import numpy as np
import json
import io
from PIL import Image
import time

app = Flask(__name__)

# Print code execute
print('Running program')

# Local paths to the model and class labels files
model_path = './models/fruit_and_vegetable_detection.h5'
labels_path = './models/metadata_arr.json'

# Load the trained model
model = load_model(model_path)
print('Model loaded')

# Load class labels
with open(labels_path, 'r') as f:
    class_labels = json.load(f)

start_time = time.time()  # Initialize start_time

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    uptime = time.time() - start_time
    uptime = round(uptime, 2)
    uptime = str(int(uptime // 3600)) + ' hours, ' + str(int((uptime % 3600) // 60)) + ' minutes, ' + str(int(uptime % 60)) + ' seconds'
    return jsonify({'status': 'UP', 'uptime': uptime})

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((300, 300))
            img_array = utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)  # Preprocess the image

            prediction = model.predict(img_array, batch_size=10)
            max_confidence = np.max(prediction)
            if max_confidence < 0.8:
                predicted_class = "Unknown"
            else:
                predicted_class = class_labels[np.argmax(prediction)]

            return jsonify({'result': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
