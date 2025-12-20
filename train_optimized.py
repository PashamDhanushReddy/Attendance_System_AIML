import cv2
import numpy as np
import mtcnn
from architecture import *
from scipy.spatial.distance import cosine
from keras.models import load_model
import pickle
import csv
import os
import random
from datetime import datetime
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import logging
from collections import Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and paths
face_data = 'Dataset'
required_shape = (160, 160)
path = "facenet_keras_weights.h5"
confidence_t = 0.99
recognition_t = 0.5

# Enhanced parameters for better accuracy
MIN_IMAGES_PER_PERSON = 5  # Minimum images required per person
FACE_DETECTION_CONFIDENCE = 0.95
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Function to normalize an image
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')

def validate_dataset():
    """Validate dataset quality and completeness"""
    logging.info("Validating dataset...")
    
    dataset_stats = {}
    valid_people = []
    
    for person_name in os.listdir(face_data):
        person_dir = os.path.join(face_data, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        images = [img for img in os.listdir(person_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_count = len(images)
        
        if image_count < MIN_IMAGES_PER_PERSON:
            logging.warning(f"{person_name} has only {image_count} images. Minimum required: {MIN_IMAGES_PER_PERSON}")
            continue
        
        valid_people.append(person_name)
        dataset_stats[person_name] = image_count
        
    logging.info(f"Dataset validation complete. Valid people: {len(valid_people)}")
    for person, count in dataset_stats.items():
        logging.info(f"  {person}: {count} images")
    
    return valid_people, dataset_stats

def enhance_face_detection(img_rgb, face_detector):
    """Enhanced face detection with multiple attempts"""
    faces = face_detector.detect_faces(img_rgb)
    
    if not faces:
        # Try with different preprocessing
        img_enhanced = cv2.equalizeHist(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
        faces = face_detector.detect_faces(img_enhanced)
    
    # Filter faces by confidence
    valid_faces = [face for face in faces if face['confidence'] > FACE_DETECTION_CONFIDENCE]
    
    return valid_faces

def extract_face_features(img_path, face_detector, face_encoder):
    """Extract face features with quality checks"""
    try:
        img_BGR = cv2.imread(img_path)
        if img_BGR is None:
            logging.warning(f"Could not read image: {img_path}")
            return None
            
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        
        # Enhanced face detection
        faces = enhance_face_detection(img_RGB, face_detector)
        
        if not faces:
            logging.warning(f"No valid faces detected in: {img_path}")
            return None
        
        # Use the best face (highest confidence)
        best_face = max(faces, key=lambda x: x['confidence'])
        
        x1, y1, width, height = best_face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Ensure face region is valid
        if x2 > img_RGB.shape[1] or y2 > img_RGB.shape[0]:
            logging.warning(f"Invalid face region in: {img_path}")
            return None
        
        face = img_RGB[y1:y2, x1:x2]
        
        # Quality checks
        if face.shape[0] < 50 or face.shape[1] < 50:
            logging.warning(f"Face too small in: {img_path}")
            return None
        
        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        
        return encode
        
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None

def train_face_recognition_optimized():
    """Optimized training with better validation and accuracy"""
    logging.info("Starting optimized face recognition training...")
    
    # Load FaceNet model
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(path)
    face_detector = mtcnn.MTCNN()
    
    # Validate dataset
    valid_people, dataset_stats = validate_dataset()
    
    if len(valid_people) < 2:
        logging.error("Need at least 2 people for training!")
        return
    
    encodes = []
    labels = []
    encoding_dict = {}
    
    # Process each person
    for idx, person_name in enumerate(valid_people):
        logging.info(f"Processing {person_name} ({idx+1}/{len(valid_people)})")
        person_dir = os.path.join(face_data, person_name)
        
        person_encodes = []
        processed_count = 0
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            encode = extract_face_features(image_path, face_detector, face_encoder)
            
            if encode is not None:
                encodes.append(encode)
                labels.append(idx)
                person_encodes.append(encode)
                processed_count += 1
        
        if person_encodes:
            encoding_dict[person_name] = person_encodes
            logging.info(f"  Extracted {processed_count} valid face encodings")
        else:
            logging.warning(f"  No valid encodings for {person_name}")
    
    if len(encodes) == 0:
        logging.error("No valid face encodings found!")
        return
    
    logging.info(f"Total encodings extracted: {len(encodes)}")
    
    # Split the data
    encodes, labels = shuffle(encodes, labels, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        encodes, labels, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Train multiple classifiers and select the best
    classifiers = {
        'SVM_linear': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
        'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'SVM_poly': SVC(kernel='poly', probability=True, degree=3, random_state=RANDOM_STATE)
    }
    
    best_classifier = None
    best_accuracy = 0
    best_name = ""
    
    logging.info("Training multiple classifiers...")
    for clf_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"{clf_name}: {accuracy:.4f} accuracy")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier
            best_name = clf_name
    
    logging.info(f"Best classifier: {best_name} with {best_accuracy:.4f} accuracy")
    
    # Detailed evaluation
    y_pred = best_classifier.predict(X_test)
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=valid_people))
    
    # Save the best classifier and encoding dictionary
    classifier_path = 'classifier/classifier_optimized.pkl'
    encoding_dict_path = 'encodings/encodings_optimized.pkl'
    stats_path = 'encodings/training_stats.json'
    
    os.makedirs('classifier', exist_ok=True)
    os.makedirs('encodings', exist_ok=True)
    
    with open(classifier_path, 'wb') as file:
        pickle.dump(best_classifier, file)
    
    with open(encoding_dict_path, 'wb') as file:
        pickle.dump(encoding_dict, file)
    
    # Save training statistics
    training_stats = {
        'best_classifier': best_name,
        'best_accuracy': float(best_accuracy),
        'total_people': len(valid_people),
        'total_encodings': len(encodes),
        'dataset_stats': dataset_stats,
        'training_date': datetime.now().isoformat()
    }
    
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logging.info("Training completed successfully!")
    logging.info(f"Model saved to: {classifier_path}")
    logging.info(f"Encodings saved to: {encoding_dict_path}")

if __name__ == "__main__":
    train_face_recognition_optimized()