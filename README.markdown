# Nail Disease Detection Using Deep Learning

This project is a web-based application that uses a deep learning model (VGG-16) to detect nail diseases from uploaded images. Built with Flask, TensorFlow, and a modern UI powered by Tailwind CSS, it provides an intuitive interface for medical professionals to classify 17 nail conditions with confidence scores.

## Project Overview

- **Purpose**: To assist in diagnosing nail diseases (e.g., Alopecia Areata, Terry's nail, Yellow nails) using image analysis.
- **Technology Stack**:
  - Backend: Flask, TensorFlow/Keras
  - Frontend: HTML, Tailwind CSS
  - Model: VGG-16 (Transfer Learning)
- **Current Date**: June 18, 2025

## Project Structure

```
nail-disease-detection/
├── static/
│   ├── models/          # Contains the trained model file (vgg-16-nail-disease.h5)
│   └── uploads/         # Temporary storage for uploaded images
├── templates/           # Contains the UI file (index.html)
├── main.py             # Flask application backend
├── train_model.py      # Script to train the VGG-16 model
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## Features

- Upload nail images via drag-and-drop or file selection.
- Real-time prediction with confidence scores.
- Responsive and modern UI with loading animations.
- Support for 17 nail disease classes.

## Installation

### Prerequisites

- Python 3.8+
- Git (for version control)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/nail-disease-detection.git
   cd nail-disease-detection
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv tf-env
   source tf-env/bin/activate  # On Windows: tf-env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**
   - Place your nail image dataset in the following structure:
     ```
     Dataset/
     ├── train/
     │   ├── Alopecia areata/
     │   ├── Beau's lines/
     │   ├── ... (other 15 classes)
     ├── test/
     │   ├── Alopecia areata/
     │   ├── Beau's lines/
     │   ├── ... (other 15 classes)
     ```
   - Update `TRAIN_PATH` and `TEST_PATH` in `train_model.py` to match your dataset location (e.g., `D:/Sak/...`).

5. **Train the Model**
   - Run the training script:
     ```bash
     python train_model.py
     ```
   - The trained model will be saved as `static/models/vgg-16-nail-disease.h5`.

6. **Run the Application**
   - Start the Flask server:
     ```bash
     python main.py
     ```
   - Open your browser and navigate to `http://localhost:5000`.

## Usage

1. **Upload an Image**
   - Drag and drop a nail image or click "Choose File" to select one.
   - The file name will be displayed for confirmation.

2. **Predict Disease**
   - Click the "Predict" button to analyze the image.
   - A loading animation will appear during processing.

3. **View Results**
   - The predicted disease and confidence score will be displayed in a card below the upload area.

## Improving Model Performance

If predictions are inaccurate (e.g., low confidence like 13.73%):
- **Enhance Dataset**: Add more images, especially for underrepresented classes (e.g., Terry's nail).
- **Adjust Training**: Modify `train_model.py` to include data augmentation (rotation, shifts) or fine-tune more VGG-16 layers.
- **Retrain**: Run `python train_model.py` with updated parameters.

## Contributing

- Fork the repository.
- Create a new branch for your feature: `git checkout -b feature-name`.
- Commit your changes: `git commit -m "Add feature"`.
- Push to the branch: `git push origin feature-name`.
- Submit a pull request.

## Acknowledgments

- Inspired by deep learning research in medical imaging.
- Utilizes TensorFlow and VGG-16 from the Keras Applications library.
