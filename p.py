# Enhanced Deepfake Detection Project

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import requests
import zipfile
import tarfile
import json
import logging
from tqdm import tqdm
import argparse
import multiprocessing
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = 224  # Standard input size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2  # Binary classification: real vs fake

# Dataset paths - will be populated in setup_datasets()
DATASET_PATHS = {
    "faceforensics": {
        "url": "https://github.com/ondyari/FaceForensics",
        "local_path": "datasets/faceforensics",
        "description": "Large-scale face forgery detection dataset"
    },
    "dfdc": {
        "url": "https://ai.meta.com/datasets/dfdc/",
        "local_path": "datasets/dfdc",
        "description": "Deepfake Detection Challenge dataset"
    },
    "celeb_df": {
        "url": "http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html",
        "local_path": "datasets/celeb_df",
        "description": "Celebrity deepfake videos dataset"
    },
    "uadfv": {
        "url": "https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi",
        "local_path": "datasets/uadfv",
        "description": "University at Albany DeepFake Videos"
    },
    "deepfake_timit": {
        "url": "https://www.idiap.ch/dataset/deepfaketimit",
        "local_path": "datasets/deepfake_timit",
        "description": "Videos of 32 subjects with high/low quality deepfakes"
    },
    "google_jigsaw": {
        "url": "https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html",
        "local_path": "datasets/google_jigsaw",
        "description": "Google/Jigsaw Deepfake Detection Dataset"
    },
    "deeper_forensics": {
        "url": "https://github.com/EndlessSora/DeeperForensics-1.0",
        "local_path": "datasets/deeper_forensics",
        "description": "Large dataset with 60,000 videos"
    },
    "dffd": {
        "url": "https://arxiv.org/abs/1910.01717",
        "local_path": "datasets/dffd",
        "description": "DeepFake Face Detection dataset"
    }
}

def setup_datasets():
    """
    Create directory structure for all datasets
    """
    # Create main datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Create directories for each dataset with real/fake subdirectories for extracted faces
    for dataset_name, dataset_info in DATASET_PATHS.items():
        dataset_path = dataset_info["local_path"]
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "raw"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "processed"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "processed", "real"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "processed", "fake"), exist_ok=True)
    
    # Create directory for the combined dataset
    os.makedirs("datasets/combined", exist_ok=True)
    os.makedirs("datasets/combined/real", exist_ok=True)
    os.makedirs("datasets/combined/fake", exist_ok=True)
    
    # Create directories for trained models and results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    logger.info("Dataset directory structure created successfully")


def download_dataset(dataset_name):
    """
    Downloads a dataset if not already present
    Note: Most deepfake datasets require manual download due to terms of use
    """
    dataset_info = DATASET_PATHS.get(dataset_name)
    if not dataset_info:
        logger.error(f"Dataset {dataset_name} not found in configuration")
        return False
    
    raw_dir = os.path.join(dataset_info["local_path"], "raw")
    
    # Check if dataset already downloaded
    if os.listdir(raw_dir):
        logger.info(f"Dataset {dataset_name} already exists at {raw_dir}")
        return True
    
    logger.info(f"Dataset '{dataset_name}' requires manual download from: {dataset_info['url']}")
    logger.info(f"Please download and place the files in: {raw_dir}")
    logger.info(f"Description: {dataset_info['description']}")
    
    return False


def extract_faces_from_video(video_path, output_dir, face_id=0, sample_rate=30, min_face_size=(64, 64)):
    """
    Extract faces from video frames and save them
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted faces
        face_id: ID for the face sequence from this video
        sample_rate: Extract a face every N frames
        min_face_size: Minimum size for a face to be considered valid
    
    Returns:
        count: Number of faces extracted
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load face detector - try different models based on availability
    try:
        # Try to use a more advanced model from DNN module
        face_detector = cv2.dnn.readNetFromCaffe(
            "models/face_detector/deploy.prototxt",
            "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        )
        use_dnn = True
        logger.info("Using DNN face detector")
    except:
        # Fall back to Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        use_dnn = False
        logger.info("Using Haar Cascade face detector")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
            
        if use_dnn:
            # DNN-based face detection
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            face_detector.setInput(blob)
            detections = face_detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:  # Threshold for face detection confidence
                    continue
                
                # Get face coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure face is large enough
                face_width, face_height = x2 - x1, y2 - y1
                if face_width < min_face_size[0] or face_height < min_face_size[1]:
                    continue
                
                # Extract face and save
                face_img = frame[y1:y2, x1:x2]
        else:
            # Haar Cascade face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=min_face_size)
            
            if len(faces) == 0:
                continue
                
            # Extract largest face (usually the main subject)
            largest_area = 0
            largest_face = None
            
            for (x, y, w, h) in faces:
                if w * h > largest_area:
                    largest_area = w * h
                    largest_face = (x, y, w, h)
                    
            if largest_face:
                x, y, w, h = largest_face
                face_img = frame[y:y+h, x:x+w]
            else:
                continue
                
        # Save face with consistent naming
        face_filename = f"{os.path.basename(video_path)}_face_{face_id}_{saved_count:04d}.jpg"
        save_path = os.path.join(output_dir, face_filename)
        
        try:
            # Resize face to consistent dimensions
            face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(save_path, face_img)
            saved_count += 1
        except Exception as e:
            logger.error(f"Error saving face image: {str(e)}")
    
    cap.release()
    return saved_count


def process_dataset(dataset_name, video_ext='.mp4', max_videos=None):
    """
    Process dataset videos to extract faces for both real and fake categories
    
    Args:
        dataset_name: Name of the dataset to process
        video_ext: Video file extension
        max_videos: Maximum number of videos to process per category (for limiting processing time)
    """
    dataset_info = DATASET_PATHS.get(dataset_name)
    if not dataset_info:
        logger.error(f"Dataset {dataset_name} not found in configuration")
        return
    
    raw_dir = os.path.join(dataset_info["local_path"], "raw")
    real_output_dir = os.path.join(dataset_info["local_path"], "processed", "real")
    fake_output_dir = os.path.join(dataset_info["local_path"], "processed", "fake")
    
    # Check if raw directory exists and contains files
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        logger.error(f"Raw data for {dataset_name} not found or directory is empty")
        return
    
    # Dataset-specific processing logic
    if dataset_name == "faceforensics":
        process_faceforensics(raw_dir, real_output_dir, fake_output_dir, max_videos)
    elif dataset_name == "dfdc":
        process_dfdc(raw_dir, real_output_dir, fake_output_dir, max_videos)
    elif dataset_name in ["celeb_df", "uadfv", "deepfake_timit"]:
        # Generic processing for datasets with real and fake folders
        process_generic_dataset(raw_dir, real_output_dir, fake_output_dir, video_ext, max_videos)
    else:
        logger.warning(f"No specific processing logic for {dataset_name}, using generic approach")
        process_generic_dataset(raw_dir, real_output_dir, fake_output_dir, video_ext, max_videos)
    
    # Combine processed images into the combined dataset
    combine_processed_faces(dataset_name)


def process_faceforensics(raw_dir, real_output_dir, fake_output_dir, max_videos):
    """
    Process FaceForensics++ dataset which has specific structure
    """
    # FaceForensics has original videos and several manipulation methods
    original_dir = os.path.join(raw_dir, "original_sequences", "youtube")
    manipulated_dir = os.path.join(raw_dir, "manipulated_sequences")
    
    # Process real videos
    real_videos_dir = os.path.join(original_dir, "c23", "videos")
    if os.path.exists(real_videos_dir):
        logger.info(f"Processing real videos from FaceForensics++")
        video_files = [f for f in os.listdir(real_videos_dir) if f.endswith(".mp4")]
        
        if max_videos:
            video_files = video_files[:max_videos]
            
        for i, video_file in enumerate(tqdm(video_files)):
            video_path = os.path.join(real_videos_dir, video_file)
            faces_extracted = extract_faces_from_video(video_path, real_output_dir, face_id=i)
            logger.info(f"Extracted {faces_extracted} faces from {video_file}")
    
    # Process fake videos (from various manipulation methods)
    manipulation_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    
    for method in manipulation_methods:
        method_dir = os.path.join(manipulated_dir, method, "c23", "videos")
        if os.path.exists(method_dir):
            logger.info(f"Processing {method} videos from FaceForensics++")
            video_files = [f for f in os.listdir(method_dir) if f.endswith(".mp4")]
            
            if max_videos:
                video_files = video_files[:max_videos]
                
            for i, video_file in enumerate(tqdm(video_files)):
                video_path = os.path.join(method_dir, video_file)
                faces_extracted = extract_faces_from_video(
                    video_path, 
                    fake_output_dir, 
                    face_id=i + 1000 * manipulation_methods.index(method)
                )
                logger.info(f"Extracted {faces_extracted} faces from {method}/{video_file}")


def process_dfdc(raw_dir, real_output_dir, fake_output_dir, max_videos):
    """
    Process DFDC dataset which has JSON metadata files
    """
    # DFDC has folders with JSON metadata
    folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f)) and f.startswith("dfdc_train_part_")]
    
    video_count = {"real": 0, "fake": 0}
    
    for folder in folders:
        folder_path = os.path.join(raw_dir, folder)
        metadata_path = os.path.join(folder_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found in {folder_path}")
            continue
        
        # Load metadata JSON
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Process videos based on metadata
        for video_filename, video_info in metadata.items():
            video_path = os.path.join(folder_path, video_filename)
            if not os.path.exists(video_path):
                continue
                
            # Check if real or fake based on metadata
            is_fake = video_info.get("label") == "FAKE"
            output_dir = fake_output_dir if is_fake else real_output_dir
            
            # Limit videos if max_videos specified
            category = "fake" if is_fake else "real"
            if max_videos and video_count[category] >= max_videos:
                continue
                
            # Extract faces from the video
            faces_extracted = extract_faces_from_video(
                video_path, 
                output_dir,
                face_id=video_count[category]
            )
            
            logger.info(f"Extracted {faces_extracted} faces from {category} video: {video_filename}")
            video_count[category] += 1


def process_generic_dataset(raw_dir, real_output_dir, fake_output_dir, video_ext, max_videos):
    """
    Generic processing for datasets with simpler structure
    """
    # Look for real and fake directories
    real_dir = None
    fake_dir = None
    
    # Try to find real/fake directories with different naming conventions
    potential_real = ["real", "Real", "original", "Original", "pristine", "Pristine", "genuine", "Genuine"]
    potential_fake = ["fake", "Fake", "deepfake", "Deepfake", "manipulated", "Manipulated", "synthetic", "Synthetic"]
    
    # First try direct subfolders
    for dirname in os.listdir(raw_dir):
        dirpath = os.path.join(raw_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
            
        if dirname in potential_real:
            real_dir = dirpath
        elif dirname in potential_fake:
            fake_dir = dirpath
    
    # If not found, search recursively
    if not real_dir or not fake_dir:
        for root, dirs, _ in os.walk(raw_dir):
            for dirname in dirs:
                if dirname in potential_real:
                    real_dir = os.path.join(root, dirname)
                elif dirname in potential_fake:
                    fake_dir = os.path.join(root, dirname)
    
    # Process real videos
    if real_dir:
        logger.info(f"Processing real videos from {real_dir}")
        process_videos_in_dir(real_dir, real_output_dir, video_ext, max_videos)
    else:
        logger.warning(f"Real videos directory not found in {raw_dir}")
    
    # Process fake videos
    if fake_dir:
        logger.info(f"Processing fake videos from {fake_dir}")
        process_videos_in_dir(fake_dir, fake_output_dir, video_ext, max_videos)
    else:
        logger.warning(f"Fake videos directory not found in {raw_dir}")


def process_videos_in_dir(video_dir, output_dir, video_ext, max_videos):
    """
    Process all videos in a directory
    """
    # Get all video files
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(video_ext):
                video_files.append(os.path.join(root, file))
    
    # Limit if max_videos specified
    if max_videos:
        video_files = video_files[:max_videos]
    
    # Process each video
    for i, video_path in enumerate(tqdm(video_files)):
        faces_extracted = extract_faces_from_video(video_path, output_dir, face_id=i)
        logger.info(f"Extracted {faces_extracted} faces from {os.path.basename(video_path)}")


def combine_processed_faces(dataset_name):
    """
    Combine processed faces from a dataset into the combined dataset directory
    """
    dataset_info = DATASET_PATHS.get(dataset_name)
    if not dataset_info:
        logger.error(f"Dataset {dataset_name} not found in configuration")
        return
    
    processed_real = os.path.join(dataset_info["local_path"], "processed", "real")
    processed_fake = os.path.join(dataset_info["local_path"], "processed", "fake")
    
    combined_real = os.path.join("datasets", "combined", "real")
    combined_fake = os.path.join("datasets", "combined", "fake")
    
    # Create combined directories if they don't exist
    os.makedirs(combined_real, exist_ok=True)
    os.makedirs(combined_fake, exist_ok=True)
    
    # Helper function to copy files with a prefix
    def copy_files_with_prefix(src_dir, dst_dir, prefix):
        if not os.path.exists(src_dir):
            logger.warning(f"Source directory {src_dir} does not exist")
            return 0
            
        count = 0
        for file in os.listdir(src_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(dst_dir, f"{prefix}_{file}")
                
                try:
                    # Using OpenCV to read, resize, and write ensures consistent format
                    img = cv2.imread(src_path)
                    if img is not None:
                        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                        cv2.imwrite(dst_path, img)
                        count += 1
                except Exception as e:
                    logger.error(f"Error copying {file}: {str(e)}")
        
        return count
    
    # Copy with dataset prefixes to avoid filename collisions
    real_count = copy_files_with_prefix(processed_real, combined_real, f"{dataset_name}_real")
    fake_count = copy_files_with_prefix(processed_fake, combined_fake, f"{dataset_name}_fake")
    
    logger.info(f"Added {real_count} real and {fake_count} fake images from {dataset_name} to combined dataset")


def build_model():
    """
    Create a convolutional neural network for deepfake detection
    using transfer learning with MobileNetV2 as the base model
    """
    # Use a pre-trained model as the base
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def create_data_generators():
    """
    Create data generators for training, validation, and testing
    """
    # Combined dataset paths
    combined_real_dir = os.path.join("datasets", "combined", "real")
    combined_fake_dir = os.path.join("datasets", "combined", "fake")
    
    # Check if directories exist and have files
    if not os.path.exists(combined_real_dir) or not os.path.exists(combined_fake_dir):
        logger.error("Combined dataset directories do not exist. Please process datasets first.")
        return None, None, None
    
    # Get all image files
    real_images = [os.path.join(combined_real_dir, f) for f in os.listdir(combined_real_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join(combined_fake_dir, f) for f in os.listdir(combined_fake_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create labels (1 for real, 0 for fake)
    real_labels = np.ones(len(real_images))
    fake_labels = np.zeros(len(fake_images))
    
    # Combine datasets
    all_images = real_images + fake_images
    all_labels = np.concatenate([real_labels, fake_labels])
    
    # Shuffle the data
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)
    all_images = np.array(all_images)[indices]
    all_labels = all_labels[indices]
    
    # Split into training, validation, and testing sets (70% train, 15% validation, 15% test)
    train_split = 0.7
    val_split = 0.15
    
    # First split train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=(1-train_split), random_state=42
    )
    
    # Then split temp into val and test
    val_test_ratio = val_split / (val_split + val_split)  # 0.5 for equal split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_test_ratio), random_state=42
    )
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and testing
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create a custom generator that can handle file paths directly
    def path_generator(paths, labels, datagen, batch_size):
        n_samples = len(paths)
        while True:
            # Shuffle at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            paths_shuffled = np.array(paths)[indices]
            labels_shuffled = labels[indices]
            
            # Generate batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_paths = paths_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                
                # Load and preprocess images
                batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3))
                for i, img_path in enumerate(batch_paths):
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                        batch_images[i] = img
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {str(e)}")
                        # Use zeros if image loading fails
                        batch_images[i] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
                
                # Apply data augmentation
                batch_images = batch_images / 255.0  # Rescale to [0,1]
                if datagen is train_datagen:
                    # Apply other augmentations from train_datagen
                    batch_images_aug = np.zeros_like(batch_images)
                    for i, img in enumerate(batch_images):
                        img = img.reshape((1,) + img.shape)
                        # Get the first (and only) augmented image
                        for aug_img in datagen.flow(img, batch_size=1):
                            batch_images_aug[i] = aug_img[0]
                            break
                    batch_images = batch_images_aug
                
                yield batch_images, batch_labels
    
    # Create generators
    train_generator = path_generator(X_train, y_train, train_datagen, BATCH_SIZE)
    val_generator = path_generator(X_val, y_val, val_datagen, BATCH_SIZE)
    test_generator = path_generator(X_test, y_test, test_datagen, BATCH_SIZE)
    
    # Create info dictionaries to mimic ImageDataGenerator's structure
    train_info = {"generator": train_generator, "steps": len(X_train) // BATCH_SIZE, "samples": len(X_train), "paths": X_train, "labels": y_train}
    val_info = {"generator": val_generator, "steps": len(X_val) // BATCH_SIZE, "samples": len(X_val), "paths": X_val, "labels": y_val}
    test_info = {"generator": test_generator, "steps": len(X_test) // BATCH_SIZE, "samples": len(X_test), "paths": X_test, "labels": y_test}
    
    return train_info, val_info, test_info


def train_model(model, train_info, val_info, output_model_path):
    """
    Train the deepfake detection model
    """
    if not train_info or not val_info:
        logger.error("Training or validation data not provided")
        return None
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        output_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_info["generator"],
        steps_per_epoch=train_info["steps"],
        epochs=EPOCHS,
        validation_data=val_info["generator"],
        validation_steps=val_info["steps"],
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join("results", "training_history.csv"), index=False)
    
    return model, history


def evaluate_model(model, test_info):
    """
    Evaluate the model and display results
    """
    if not test_info:
        logger.error("Test data not provided")
        return None
    
    # Evaluate the model
    logger.info("Evaluating model on test data...")
    evaluation = model.evaluate(
        test_info["generator"],
        steps=test_info["steps"]
    )
    
    # Create a dictionary of metrics
    metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = evaluation[i]
        logger.info(f"{metric_name}: {evaluation[i]:.4f}")
    
    # Generate predictions for confusion matrix and ROC curve
    logger.info("Generating predictions for detailed analysis...")
    y_pred_prob = []
    y_true = []
    
    # Use a fraction of the test set for analysis to avoid memory issues
    num_samples = min(1000, test_info["samples"])
    indices = np.random.choice(range(test_info["samples"]), num_samples, replace=False)
    
    # Get selected images and labels
    selected_paths = [test_info["paths"][i] for i in indices]
    selected_labels = test_info["labels"][indices]
    
    # Predict on selected images
    for img_path in tqdm(selected_paths, desc="Predicting"):
        try:
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Get prediction
            pred = model.predict(img, verbose=0)[0][0]
            y_pred_prob.append(pred)
        except Exception as e:
            logger.error(f"Error predicting image {img_path}: {str(e)}")
            # Use a default prediction for errors
            y_pred_prob.append(0.5)
    
    y_true = selected_labels
    y_pred_classes = (np.array(y_pred_prob) > 0.5).astype(int)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Print classification report
    report = classification_report(y_true, y_pred_classes)
    logger.info("Classification Report:")
    logger.info("\n" + report)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Fake', 'Real']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join("results", "confusion_matrix.png"))
    
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("results", "roc_curve.png"))
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    metrics_df.to_csv(os.path.join("results", "evaluation_metrics.csv"), index=False)
    
    return metrics, (fpr, tpr, roc_auc)


def predict_single_image(model, image_path):
    """
    Predict whether a single image is real or fake
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)[0][0]
        
        # Determine result and confidence
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        logger.info(f"Prediction for {image_path}: {result} (Confidence: {confidence:.2f})")
        return result, confidence
    
    except Exception as e:
        logger.error(f"Error predicting image {image_path}: {str(e)}")
        return None, None


def visualize_attention(model, image_path, output_path):
    """
    Generate a heatmap visualization showing which parts of the image
    the model focuses on for its prediction
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        
        # Create a model that outputs both the prediction and the last conv layer
        last_conv_layer = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                last_conv_idx = i
        
        if last_conv_layer is None:
            logger.error("Could not find convolutional layer in model")
            return False
        
        # Extract features and make prediction
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            class_idx = 1 if predictions[0][0] > 0.5 else 0
            output = predictions[:, 0]
        
        # Get gradients of the output with respect to the last conv layer
        grads = tape.gradient(output, conv_output)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by their gradient importance
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * 0.4 + img
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Get prediction result
        result = "Real" if predictions[0][0] > 0.5 else "Fake"
        confidence = float(predictions[0][0]) if predictions[0][0] > 0.5 else float(1 - predictions[0][0])
        
        # Add text with prediction info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{result} (Confidence: {confidence:.2f})"
        cv2.putText(superimposed_img, text, (10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Save the visualization
        cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved attention visualization to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating attention map: {str(e)}")
        return False


def predict_on_video(model, video_path, output_path, sample_rate=1):
    """
    Run deepfake detection on a video, mark each frame with prediction result,
    and save the annotated video
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a face detector
        try:
            face_detector = cv2.dnn.readNetFromCaffe(
                "models/face_detector/deploy.prototxt",
                "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
            )
            use_dnn = True
        except:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            use_dnn = False
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frames
        frame_idx = 0
        
        # For tracking prediction smoothing
        recent_predictions = []
        smoothing_window = 5
        
        with tqdm(total=frame_count, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                pbar.update(1)
                
                # Process every Nth frame (based on sample_rate)
                if frame_idx % sample_rate != 0:
                    out.write(frame)
                    continue
                
                # Save original frame for output
                output_frame = frame.copy()
                
                # Detect faces
                if use_dnn:
                    # DNN-based face detection
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 
                        1.0, (300, 300), 
                        (104.0, 177.0, 123.0)
                    )
                    face_detector.setInput(blob)
                    detections = face_detector.forward()
                    
                    faces = []
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence < 0.5:  # Threshold for face detection confidence
                            continue
                        
                        # Get face coordinates
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (x1, y1, x2, y2) = box.astype("int")
                        faces.append((x1, y1, x2-x1, y2-y1))
                else:
                    # Haar Cascade face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Process each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face
                    if use_dnn:
                        face_img = frame[y:y+h, x:x+w]
                    else:
                        face_img = frame[y:y+h, x:x+w]
                    
                    # Ensure face is large enough
                    if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                        continue
                    
                    # Preprocess face for prediction
                    try:
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                        face_normalized = face_resized / 255.0
                        face_batch = np.expand_dims(face_normalized, axis=0)
                        
                        # Make prediction
                        prediction = float(model.predict(face_batch, verbose=0)[0][0])
                        
                        # Smooth predictions over time
                        recent_predictions.append(prediction)
                        if len(recent_predictions) > smoothing_window:
                            recent_predictions.pop(0)
                        
                        smoothed_prediction = np.mean(recent_predictions)
                        
                        # Determine result and confidence
                        result = "Real" if smoothed_prediction > 0.5 else "Fake"
                        confidence = smoothed_prediction if smoothed_prediction > 0.5 else 1 - smoothed_prediction
                        
                        # Draw rectangle around face
                        color = (0, 255, 0) if result == "Real" else (0, 0, 255)  # Green for real, red for fake
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Add text with prediction
                        text = f"{result}: {confidence:.2f}"
                        cv2.putText(output_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    except Exception as e:
                        logger.error(f"Error processing face in frame {frame_idx}: {str(e)}")
                
                # Write the frame with annotations
                out.write(output_frame)
        
        # Release resources
        cap.release()
        out.release()
        logger.info(f"Processed video saved to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return False


def download_face_detector():
    """
    Download pre-trained face detector model if not already present
    """
    models_dir = "models/face_detector"
    os.makedirs(models_dir, exist_ok=True)
    
    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Check if files already exist
    if os.path.exists(prototxt_path) and os.path.exists(model_path):
        logger.info("Face detector model already downloaded")
        return True
    
    # URLs for the face detector model files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    try:
        # Download prototxt
        logger.info("Downloading face detector prototxt...")
        response = requests.get(prototxt_url)
        with open(prototxt_path, "wb") as f:
            f.write(response.content)
        
        # Download model
        logger.info("Downloading face detector model...")
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        logger.info("Face detector model downloaded successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading face detector model: {str(e)}")
        logger.info("Using Haar Cascade face detector as fallback")
        return False


def create_dataset_stats():
    """
    Generate statistics about the processed datasets
    """
    combined_real_dir = os.path.join("datasets", "combined", "real")
    combined_fake_dir = os.path.join("datasets", "combined", "fake")
    
    if not os.path.exists(combined_real_dir) or not os.path.exists(combined_fake_dir):
        logger.error("Combined dataset directories not found")
        return
    
    # Count images by dataset
    dataset_counts = {}
    
    # Count real images by dataset
    for filename in os.listdir(combined_real_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        dataset_name = filename.split("_")[0]
        if dataset_name not in dataset_counts:
            dataset_counts[dataset_name] = {"real": 0, "fake": 0}
        
        dataset_counts[dataset_name]["real"] += 1
    
    # Count fake images by dataset
    for filename in os.listdir(combined_fake_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        dataset_name = filename.split("_")[0]
        if dataset_name not in dataset_counts:
            dataset_counts[dataset_name] = {"real": 0, "fake": 0}
        
        dataset_counts[dataset_name]["fake"] += 1
    
    # Calculate totals
    total_real = sum(dataset["real"] for dataset in dataset_counts.values())
    total_fake = sum(dataset["fake"] for dataset in dataset_counts.values())
    total_images = total_real + total_fake
    
    # Create stats DataFrame
    stats = []
    for dataset_name, counts in dataset_counts.items():
        stats.append({
            "Dataset": dataset_name,
            "Real Images": counts["real"],
            "Fake Images": counts["fake"],
            "Total Images": counts["real"] + counts["fake"],
            "Real %": round(counts["real"] * 100 / (counts["real"] + counts["fake"]), 2) if (counts["real"] + counts["fake"]) > 0 else 0,
            "Fake %": round(counts["fake"] * 100 / (counts["real"] + counts["fake"]), 2) if (counts["real"] + counts["fake"]) > 0 else 0
        })
    
    # Add total row
    stats.append({
        "Dataset": "TOTAL",
        "Real Images": total_real,
        "Fake Images": total_fake,
        "Total Images": total_images,
        "Real %": round(total_real * 100 / total_images, 2) if total_images > 0 else 0,
        "Fake %": round(total_fake * 100 / total_images, 2) if total_images > 0 else 0
    })
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join("results", "dataset_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    
    # Print statistics
    logger.info(f"Dataset statistics saved to {stats_path}")
    logger.info(f"Total images: {total_images} ({total_real} real, {total_fake} fake)")
    
    return stats_df


def main():
    """
    Main function to run the deepfake detection project
    """
    parser = argparse.ArgumentParser(description="Enhanced Deepfake Detection Project")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up directory structure and download resources")
    
    # Download dataset command
    download_parser = subparsers.add_parser("download", help="Show instructions for downloading datasets")
    download_parser.add_argument("--dataset", type=str, choices=DATASET_PATHS.keys(), help="Dataset to download")
    
    # Process dataset command
    process_parser = subparsers.add_parser("process", help="Process a dataset to extract faces")
    process_parser.add_argument("--dataset", type=str, required=True, choices=DATASET_PATHS.keys(), help="Dataset to process")
    process_parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process per category")
    
    # Process all datasets command
    process_all_parser = subparsers.add_parser("process-all", help="Process all available datasets")
    process_all_parser.add_argument("--max-videos", type=int, default=50, help="Maximum number of videos to process per category and dataset")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train the deepfake detection model")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--model-path", type=str, default="models/deepfake_detector.h5", help="Path to save the trained model")
    
    # Evaluate model command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    eval_parser.add_argument("--model-path", type=str, default="models/deepfake_detector.h5", help="Path to the trained model")
    
    # Predict on image command
    predict_img_parser = subparsers.add_parser("predict-image", help="Predict on a single image")
    predict_img_parser.add_argument("--model-path", type=str, default="models/deepfake_detector.h5", help="Path to the trained model")
    predict_img_parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    predict_img_parser.add_argument("--visualize", action="store_true", help="Visualize attention heatmap")
    
    # Predict on video command
    predict_vid_parser = subparsers.add_parser("predict-video", help="Predict on a video")
    predict_vid_parser.add_argument("--model-path", type=str, default="models/deepfake_detector.h5", help="Path to the trained model")
    predict_vid_parser.add_argument("--video-path", type=str, required=True, help="Path to the input video")
    predict_vid_parser.add_argument("--output-path", type=str, default=None, help="Path to save the output video")
    predict_vid_parser.add_argument("--sample-rate", type=int, default=30, help="Process every Nth frame")
    
    # Dataset statistics command
    stats_parser = subparsers.add_parser("stats", help="Generate dataset statistics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "setup":
        logger.info("Setting up directory structure...")
        setup_datasets()
        download_face_detector()
        
    elif args.command == "download":
        if args.dataset:
            download_dataset(args.dataset)
        else:
            for dataset_name in DATASET_PATHS:
                download_dataset(dataset_name)
    
    elif args.command == "process":
        logger.info(f"Processing dataset: {args.dataset}")
        process_dataset(args.dataset, max_videos=args.max_videos)
        
    elif args.command == "process-all":
        for dataset_name in DATASET_PATHS:
            logger.info(f"Processing dataset: {dataset_name}")
            process_dataset(dataset_name, max_videos=args.max_videos)
        
    elif args.command == "train":
        # Update global configuration
        global EPOCHS, BATCH_SIZE
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size
        
        logger.info(f"Training model with {EPOCHS} epochs and batch size {BATCH_SIZE}")
        
        # Create data generators
        train_info, val_info, test_info = create_data_generators()
        
        if not train_info or not val_info or not test_info:
            logger.error("Failed to create data generators")
            return
        
        # Build and train model
        model = build_model()
        model, history = train_model(model, train_info, val_info, args.model_path)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", "training_history.png"))
        
    elif args.command == "evaluate":
        logger.info(f"Evaluating model: {args.model_path}")
        
        # Load model
        try:
            model = load_model(args.model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return
        
        # Create data generators
        _, _, test_info = create_data_generators()
        
        if not test_info:
            logger.error("Failed to create test data generator")
            return
        
        # Evaluate model
        metrics, roc_data = evaluate_model(model, test_info)
        
    elif args.command == "predict-image":
        logger.info(f"Predicting on image: {args.image_path}")
        
        # Check if image exists
        if not os.path.exists(args.image_path):
            logger.error(f"Image not found: {args.image_path}")
            return
        
        # Load model
        try:
            model = load_model(args.model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return
        
        # Predict on image
        result, confidence = predict_single_image(model, args.image_path)
        
        if result is None:
            logger.error("Prediction failed")
            return
        
        logger.info(f"Prediction: {result} (Confidence: {confidence:.2f})")
        
        # Visualize attention if requested
        if args.visualize:
            output_path = os.path.splitext(args.image_path)[0] + "_attention.jpg"
            visualize_attention(model, args.image_path, output_path)
        
    elif args.command == "predict-video":
        logger.info(f"Predicting on video: {args.video_path}")
        
        # Check if video exists
        if not os.path.exists(args.video_path):
            logger.error(f"Video not found: {args.video_path}")
            return
        
        # Load model
        try:
            model = load_model(args.model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return
        
        # Set output path if not provided
        output_path = args.output_path
        if output_path is None:
            output_path = os.path.splitext(args.video_path)[0] + "_predicted.mp4"
        
        # Predict on video
        predict_on_video(model, args.video_path, output_path, sample_rate=args.sample_rate)
        
    elif args.command == "stats":
        logger.info("Generating dataset statistics...")
        create_dataset_stats()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()