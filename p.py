# Deepfake Detection Project

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# Configuration
IMAGE_SIZE = 224  # Standard input size for many pre-trained models
BATCH_SIZE = 32
EPOCHS = 10

def build_model():
    """
    Create a convolutional neural network for deepfake detection
    using transfer learning with MobileNetV2 as the base model
    """
    # Use a pre-trained model as the base
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            include_top=False,
                            weights='imagenet')
    base_model.trainable = False  # Freeze the pre-trained layers
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification (real vs fake)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for the model"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0  # Normalize pixel values
    return image

def extract_faces(video_path, output_dir, sample_rate=1):
    """
    Extract faces from video frames and save them
    sample_rate: extract a face every N frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every N frames (based on sample_rate)
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Extract the face with some margin
            face_img = frame[y:y+h, x:x+w]
            
            # Save the face
            save_path = os.path.join(output_dir, f'face_{saved_count}.jpg')
            cv2.imwrite(save_path, face_img)
            saved_count += 1
            
    cap.release()
    return saved_count

def load_dataset(real_dir, fake_dir):
    """Load and prepare the dataset of real and fake images"""
    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
    
    # Create labels (0 for fake, 1 for real)
    real_labels = np.ones(len(real_images))
    fake_labels = np.zeros(len(fake_images))
    
    # Combine datasets
    all_images = real_images + fake_images
    all_labels = np.concatenate([real_labels, fake_labels])
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, X_test, y_train, y_test):
    """Train the deepfake detection model"""
    # Data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=lambda img: img / 255.0
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=lambda img: img / 255.0
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        X_train,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    test_generator = test_datagen.flow_from_directory(
        X_test,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS
    )
    
    return model, history

def evaluate_model(model, test_generator):
    """Evaluate the model and display results"""
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_generator.classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Print classification report
    report = classification_report(test_generator.classes, y_pred_classes)
    print("Classification Report:")
    print(report)
    
    return y_pred

def predict_image(model, image_path):
    """Predict whether an image is real or fake"""
    # Preprocess the image
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)[0][0]
    
    result = "Real" if prediction > 0.5 else "Fake"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"Prediction: {result} (Confidence: {confidence:.2f})")
    return result, confidence

def main():
    """Main function to run the deepfake detection project"""
    # Define directories for dataset
    real_dir = "path/to/real/images"
    fake_dir = "path/to/fake/images"
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(real_dir, fake_dir)
    
    # Build model
    model = build_model()
    
    # Train model
    model, history = train_model(model, X_train, X_test, y_train, y_test)
    
    # Save the trained model
    model.save("deepfake_detector.h5")
    
    # Test on new images
    test_image = "path/to/test/image.jpg"
    result, confidence = predict_image(model, test_image)

if __name__ == "__main__":
    main()