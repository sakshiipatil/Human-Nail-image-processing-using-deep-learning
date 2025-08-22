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
CORS(app)  # Enable CORS for local development

# Define paths
MODEL_PATH = os.path.join('app','static', 'models', 'vgg-16-nail-disease.h5')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
class_labels = [
    "Darier's disease",
    "Muehrcke's lines",
    "Alopecia areata",
    "Beau's lines",
    "Bluish nail",
    "Clubbing",
    "Eczema",
    "Half and half nails (Lindsay's nails)",
    "Koilonychia",
    "Leukonychia",
    "Onycholysis",
    "Pale nail",
    "Red lunula",
    "Splinter hemorrhage",
    "Terry's nail",
    "White nail",
    "Yellow nails"
]

# Load the model with error handling
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully. Input shape: {model.input_shape}, Output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Ensure upload directory exists
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload directory ensured at: {UPLOAD_FOLDER}")
except Exception as e:
    logger.error(f"Failed to create upload directory: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is provided
        if 'file' not in request.files:
            logger.warning("No file uploaded in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty file name in request")
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logger.info(f"Saving uploaded file to: {img_path}")
        try:
            file.save(img_path)
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            return jsonify({'error': f"Failed to save file: {str(e)}"}), 500

        # Load and preprocess the image
        logger.info(f"Loading and preprocessing image: {img_path}")
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            logger.info(f"Image preprocessed. Shape: {img_array.shape}")
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return jsonify({'error': f"Image processing failed: {str(e)}"}), 400

        # Verify model output compatibility
        if model.output_shape[1] != len(class_labels):
            logger.error(f"Model output shape {model.output_shape[1]} does not match class labels count {len(class_labels)}")
            return jsonify({'error': 'Model output does not match class labels'}), 500

        # Make prediction
        logger.info("Running model prediction")
        try:
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])
            logger.info(f"Prediction successful. Predicted class: {predicted_class}, Confidence: {confidence}")
            return jsonify({
                'disease_name': predicted_class,
                'confidence': round(confidence * 100, 2)
            })
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)