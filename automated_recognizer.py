import cv2
import numpy as np
import mtcnn
from architecture import *
from scipy.spatial.distance import cosine
from keras.models import load_model
import pickle
import csv
import os
from datetime import datetime
import logging
import time
from collections import deque, Counter
from sklearn.preprocessing import Normalizer
import threading
import tkinter as tk
from tkinter import messagebox

# Define normalize function and l2_normalizer
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced parameters
confidence_t = 0.95  # Increased confidence threshold
recognition_t = 0.4   # Decreased recognition threshold for better accuracy
required_size = (160, 160)

# Recognition smoothing parameters
SMOOTHING_WINDOW = 5  # Number of frames to average
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence for recognition
FACE_DETECTION_SKIP = 2  # Process every Nth frame for performance

class AutomatedFaceRecognitionSystem:
    def __init__(self):
        self.face_encoder = None
        self.face_detector = None
        self.encoding_dict = None
        self.classifier = None
        self.name_history = deque(maxlen=SMOOTHING_WINDOW)
        self.frame_count = 0
        self.last_recognition_time = {}
        self.recognition_cooldown = 1  # seconds between recognitions (reduced for testing)
        self.detected_faces = set()  # Track detected faces
        self.max_detection_time = 10  # Maximum time to run (seconds)
        self.start_time = None
        self.camera_window = None
        self.root = None
        
    def load_models(self):
        """Load all required models"""
        logging.info("Loading models...")
        
        # Load face encoder
        self.face_encoder = InceptionResNetV2()
        self.face_encoder.load_weights("facenet_keras_weights.h5")
        
        # Load face detector
        self.face_detector = mtcnn.MTCNN()
        
        # Load encoding dictionary
        try:
            with open('encodings/encodings_optimized.pkl', 'rb') as f:
                self.encoding_dict = pickle.load(f)
        except FileNotFoundError:
            logging.warning("Optimized encodings not found, using standard encodings")
            with open('encodings/encodings.pkl', 'rb') as f:
                self.encoding_dict = pickle.load(f)
        
        # Load classifier
        try:
            with open('classifier/classifier_optimized.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
        except FileNotFoundError:
            logging.warning("Optimized classifier not found, using standard classifier")
            with open('classifier/classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
        
        logging.info("Models loaded successfully")
        
    def get_face(self, img, box):
        """Extract face from bounding box with validation"""
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Validate coordinates
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None, None
            
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)
    
    def get_encode(self, face, size):
        """Get face encoding with quality checks"""
        if face is None or face.size == 0:
            return None
            
        # Check minimum face size
        if face.shape[0] < 50 or face.shape[1] < 50:
            return None
        
        try:
            face = normalize(face)
            face = cv2.resize(face, size)
            encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
            return encode.flatten()
        except Exception as e:
            logging.error(f"Error encoding face: {e}")
            return None
    
    def smooth_recognition(self, name):
        """Apply smoothing to recognition results"""
        self.name_history.append(name)
        
        # Count occurrences of each name
        name_counts = Counter(self.name_history)
        
        # Return the most common name if it appears in majority
        if len(self.name_history) >= 3:
            most_common = name_counts.most_common(1)[0]
            if most_common[1] >= len(self.name_history) * 0.6:  # 60% majority
                return most_common[0]
        
        return name
    
    def should_recognize(self, name):
        """Check if we should recognize this person again"""
        current_time = time.time()
        
        if name in self.last_recognition_time:
            time_since_last = current_time - self.last_recognition_time[name]
            if time_since_last < self.recognition_cooldown:
                logging.info(f"Skipping {name} - cooldown active ({time_since_last:.1f}s < {self.recognition_cooldown}s)")
                return False
        
        self.last_recognition_time[name] = current_time
        logging.info(f"Recognition allowed for {name}")
        return True
    
    def mark_attendance(self, name):
        """Enhanced attendance marking with validation"""
        if name == 'unknown' or not self.should_recognize(name):
            return False
        
        now = datetime.now()
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%I:%M %p")
        
        file_name = f"att_{current_date}.csv"
        
        try:
            # Read existing data
            existing_data = []
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    existing_data = [line.strip().split(',') for line in f.readlines()]
            
            # Check if person already marked present
            person_found = False
            for i, entry in enumerate(existing_data):
                if len(entry) >= 1 and entry[0] == name:
                    # Update exit time
                    if len(entry) >= 4:
                        existing_data[i][3] = current_time
                    person_found = True
                    break
            
            # Add new entry if not found
            if not person_found:
                new_entry = [name, current_date, current_time, current_time, "Present"]
                existing_data.append(new_entry)
                logging.info(f"Attendance marked for {name}")
            
            # Write back to file
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                # Always write header for proper format
                writer.writerow(["Name", "Date", "EntryTime", "ExitTime", "Status"])
                writer.writerows(existing_data)
                
            return True  # Successfully marked attendance
                
        except Exception as e:
            logging.error(f"Error marking attendance for {name}: {e}")
            return False
    
    def recognize_face(self, img_rgb, face_box):
        """Enhanced face recognition with multiple methods"""
        face, pt_1, pt_2 = self.get_face(img_rgb, face_box['box'])
        
        if face is None:
            return 'unknown', 1.0, pt_1, pt_2
        
        # Get face encoding
        encode = self.get_encode(face, required_size)
        if encode is None:
            return 'unknown', 1.0, pt_1, pt_2
        
        # Normalize encoding
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        
        # Method 1: SVM Classifier
        try:
            svm_prediction = self.classifier.predict([encode])[0]
            svm_proba = self.classifier.predict_proba([encode])[0]
            svm_confidence = max(svm_proba)
            
            if svm_confidence > CONFIDENCE_THRESHOLD:
                person_name = list(self.encoding_dict.keys())[svm_prediction]
                return person_name, 1 - svm_confidence, pt_1, pt_2
        except Exception as e:
            logging.error(f"SVM prediction error: {e}")
        
        # Method 2: Cosine similarity with encoding dictionary
        best_match = 'unknown'
        best_distance = float('inf')
        
        for db_name, db_encodes in self.encoding_dict.items():
            for db_encode in db_encodes:
                try:
                    dist = cosine(db_encode.flatten(), encode.flatten())
                    if dist < recognition_t and dist < best_distance:
                        best_match = db_name
                        best_distance = dist
                except Exception as e:
                    continue
        
        return best_match, best_distance, pt_1, pt_2
    
    def create_camera_window(self):
        """Create a simple camera window"""
        self.root = tk.Tk()
        self.root.title("Face Recognition - Attendance System")
        self.root.geometry("400x100")
        self.root.configure(bg='#f0f0f0')
        
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create status label
        self.status_label = tk.Label(self.root, text="Initializing camera...", 
                                   font=("Arial", 12), bg='#f0f0f0')
        self.status_label.pack(pady=20)
        
        # Create detected faces label
        self.faces_label = tk.Label(self.root, text="Detected: None", 
                                  font=("Arial", 10), bg='#f0f0f0')
        self.faces_label.pack()
        
        self.root.update()
    
    def update_status(self, message):
        """Update the status label"""
        if self.status_label:
            self.status_label.config(text=message)
            self.root.update()
    
    def update_detected_faces(self, faces):
        """Update the detected faces label"""
        if self.faces_label:
            if faces:
                self.faces_label.config(text=f"Detected: {', '.join(faces)}")
            else:
                self.faces_label.config(text="Detected: None")
            self.root.update()
    
    def run_automated(self):
        """Automated face recognition that opens camera, detects faces, marks attendance, and closes"""
        logging.info("Starting automated face recognition system...")
        
        self.load_models()
        
        # Create camera window
        self.create_camera_window()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Cannot open camera")
            self.update_status("Error: Cannot open camera")
            if self.root:
                self.root.after(2000, self.root.destroy)
            return
        
        logging.info("Camera opened successfully")
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logging.error("Cannot read frame from camera")
                    self.update_status("Error: Cannot read frame")
                    break
                
                # Process frame
                processed_frame = self.recognize_automated(frame)
                
                # Check if we've detected enough faces or time limit reached
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                if len(self.detected_faces) >= 1 or elapsed_time >= self.max_detection_time:
                    if len(self.detected_faces) >= 1:
                        self.update_status(f"Detected {len(self.detected_faces)} faces. Marking attendance...")
                        time.sleep(2)  # Give user time to see the result
                    else:
                        self.update_status("No faces detected. Closing...")
                        time.sleep(2)
                    break
                
                # Show camera feed in a separate window
                cv2.imshow('Face Recognition Camera', processed_frame)
                
                # Check for manual exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logging.error(f"Error during automated execution: {e}")
            self.update_status(f"Error: {str(e)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.root:
                self.root.destroy()
            
            # Show final results
            self.show_final_results()
            
            logging.info("Automated face recognition system stopped")
    
    def recognize_automated(self, img):
        """Automated recognition function"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % FACE_DETECTION_SKIP != 0:
            return img
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)
        
        current_faces = set()
        
        for res in results:
            if res['confidence'] < confidence_t:
                continue
            
            name, distance, pt_1, pt_2 = self.recognize_face(img_rgb, res)
            
            # Apply smoothing
            smoothed_name = self.smooth_recognition(name)
            
            if smoothed_name != 'unknown':
                current_faces.add(smoothed_name)
                logging.info(f"Detected face: {smoothed_name}, confidence: {1-distance:.2f}")
                attendance_marked = self.mark_attendance(smoothed_name)
                
                if attendance_marked:
                    self.detected_faces.add(smoothed_name)
                    self.update_status(f"Marked attendance for {smoothed_name}")
                    logging.info(f"Successfully marked attendance for {smoothed_name}")
                else:
                    logging.warning(f"Failed to mark attendance for {smoothed_name}")
            else:
                logging.info(f"Unknown face detected, confidence: {1-distance:.2f}")
            
            # Draw rectangle and name
            color = (0, 255, 0) if smoothed_name != 'unknown' else (0, 0, 255)
            cv2.rectangle(img, pt_1, pt_2, color, 2)
            
            label = f"{smoothed_name} ({1-distance:.2f})"
            cv2.putText(img, label, (pt_1[0], pt_1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Update detected faces in GUI
        self.update_detected_faces(list(current_faces))
        
        # Add system info overlay
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        cv2.putText(img, f"Time: {elapsed_time:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Detected: {len(self.detected_faces)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def show_final_results(self):
        """Show final results in a message box"""
        if self.detected_faces:
            message = f"Successfully marked attendance for:\n\n"
            for name in sorted(self.detected_faces):
                message += f"• {name}\n"
            message += f"\nTotal: {len(self.detected_faces)} students"
            messagebox.showinfo("Attendance Marked Successfully!", message)
        else:
            messagebox.showwarning("No Faces Detected", 
                                 "No faces were detected during the recognition process.\n\n"
                                 "Please ensure:\n"
                                 "• Good lighting conditions\n"
                                 "• Face is clearly visible\n"
                                 "• Camera is working properly\n\n"
                                 "Try again in a few seconds.")
    
    def run(self):
        """Main execution function"""
        self.run_automated()

def main():
    """Main function"""
    system = AutomatedFaceRecognitionSystem()
    system.run()

if __name__ == "__main__":
    main()