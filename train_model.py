import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam
from glob import glob
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define image size and paths
IMAGE_SIZE = [224, 224]
TRAIN_PATH = 'D:/Sak/Human-nail-image-processing-using-deep-learning/Dataset/train'
TEST_PATH = 'D:/Sak/Human-nail-image-processing-using-deep-learning/Dataset/test'
MODEL_PATH = 'static/models/vgg-16-nail-disease.h5'

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Class labels
class_labels = [
    "Darier's disease", "Muehrcke's lines", "Alopecia areata", "Beau's lines",
    "Bluish nail", "Clubbing", "Eczema", "Half and half nails (Lindsay's nails)",
    "Koilonychia", "Leukonychia", "Onycholysis", "Pale nail", "Red lunula",
    "Splinter hemorrhage", "Terry's nail", "White nail", "Yellow nails"
]

try:
    # Load VGG16 base model
    logger.info("Loading VGG16 base model...")
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    for layer in vgg.layers[-4:]:  # Fine-tune last 4 layers
        layer.trainable = True

    # Add custom layers
    logger.info("Adding custom layers...")
    x = Flatten()(vgg.output)
    x = Dropout(0.5)(x)
    prediction = Dense(len(class_labels), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    # Compile the model
    logger.info("Compiling model...")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    # Data augmentation and generators
    logger.info("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Train the model
    logger.info("Starting model training...")
    history = model.fit(
        train_set,
        validation_data=test_set,
        epochs=150,
        steps_per_epoch=len(train_set) // 3,
        validation_steps=len(test_set) // 3
    )

    # Save the model
    logger.info(f"Saving model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    logger.info("Model training and saving completed successfully!")

except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise