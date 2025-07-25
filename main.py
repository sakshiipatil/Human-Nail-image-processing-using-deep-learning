from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Define paths
MODEL_PATH = os.path.join('static', 'models', 'vgg-16-nail-disease.h5')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
class_labels = [
    "Darier's disease", "Muehrcke's lines", "Alopecia areata", "Beau's lines",
    "Bluish nail", "Clubbing", "Eczema", "Half and half nails (Lindsay's nails)",
    "Koilonychia", "Leukonychia", "Onycholysis", "Pale nail", "Red lunula",
    "Splinter hemorrhage", "Terry's nail", "White nail", "Yellow nails"
]

# Load the model with error handling
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        return jsonify({
            'disease_name': predicted_class,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)