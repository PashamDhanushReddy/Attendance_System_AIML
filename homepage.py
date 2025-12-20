from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import threading
import time
from collections import deque, Counter
from datetime import datetime
import csv
import numpy as np
import mtcnn
from architecture import *
from scipy.spatial.distance import cosine
from keras.models import load_model
import pickle
from sklearn.preprocessing import Normalizer

# Define normalize function and l2_normalizer
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')

# Global instance of FaceRecognitionSystem
face_recognition_system = None

# Enhanced parameters
confidence_t = 0.95  # Increased confidence threshold
recognition_t = 0.4   # Decreased recognition threshold for better accuracy
required_size = (160, 160)

# Recognition smoothing parameters
SMOOTHING_WINDOW = 5  # Number of frames to average
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence for recognition
FACE_DETECTION_SKIP = 2  # Process every Nth frame for performance

class FaceRecognitionSystem:
    def __init__(self, camera_frame_label, root_window):
        self.face_encoder = None
        self.face_detector = None
        self.encoding_dict = None
        self.classifier = None
        self.name_history = deque(maxlen=SMOOTHING_WINDOW)
        self.frame_count = 0
        self.last_recognition_time = {}
        self.recognition_cooldown = 1  # seconds between recognitions
        self.detected_faces = set()  # Track detected faces
        self.max_detection_time = 10  # Maximum time to run (seconds)
        self.start_time = None
        self.is_running = False
        self.cap = None
        self.camera_frame_label = camera_frame_label
        self.root_window = root_window
        self.current_image = None
        self.load_models()

    def load_models(self):
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
            with open('encodings/encodings.pkl', 'rb') as f:
                self.encoding_dict = pickle.load(f)
        
        # Load classifier
        try:
            with open('classifier/classifier_optimized.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
        except FileNotFoundError:
            with open('classifier/classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)

    def get_face(self, img, box):
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None, None
            
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)
    
    def get_encode(self, face, size):
        if face is None or face.size == 0:
            return None
            
        if face.shape[0] < 50 or face.shape[1] < 50:
            return None
        
        try:
            face = normalize(face)
            face = cv2.resize(face, size)
            encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
            return encode.flatten()
        except Exception as e:
            return None
    
    def smooth_recognition(self, name):
        self.name_history.append(name)
        name_counts = Counter(self.name_history)
        
        if len(self.name_history) >= 3:
            most_common = name_counts.most_common(1)[0]
            if most_common[1] >= len(self.name_history) * 0.6:
                return most_common[0]
        
        return name
    
    def should_recognize(self, name):
        current_time = time.time()
        
        if name in self.last_recognition_time:
            time_since_last = current_time - self.last_recognition_time[name]
            if time_since_last < self.recognition_cooldown:
                return False
        
        self.last_recognition_time[name] = current_time
        return True
    
    def mark_attendance(self, name):
        if name == 'unknown' or not self.should_recognize(name):
            return False
        
        now = datetime.now()
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%I:%M %p")
        
        file_name = f"att_{current_date}.csv"
        
        try:
            existing_data = []
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    existing_data = [line.strip().split(',') for line in f.readlines()]
            
            person_found = False
            for i, entry in enumerate(existing_data):
                if len(entry) >= 1 and entry[0] == name:
                    if len(entry) >= 4:
                        existing_data[i][3] = current_time
                    person_found = True
                    break
            
            if not person_found:
                new_entry = [name, current_date, current_time, current_time, "Present"]
                existing_data.append(new_entry)
            
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "EntryTime", "ExitTime", "Status"])
                writer.writerows(existing_data)
                
            return True
                
        except Exception as e:
            messagebox.showerror("Attendance Error", f"Error marking attendance for {name}: {e}")
            return False
    
    def recognize_face(self, img_rgb, face_box):
        face, pt_1, pt_2 = self.get_face(img_rgb, face_box['box'])
        
        if face is None:
            return 'unknown', 1.0, pt_1, pt_2
        
        encode = self.get_encode(face, required_size)
        if encode is None:
            return 'unknown', 1.0, pt_1, pt_2
        
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        
        try:
            svm_prediction = self.classifier.predict([encode])[0]
            svm_proba = self.classifier.predict_proba([encode])[0]
            svm_confidence = max(svm_proba)
            
            if svm_confidence > CONFIDENCE_THRESHOLD:
                person_name = list(self.encoding_dict.keys())[svm_prediction]
                return person_name, 1 - svm_confidence, pt_1, pt_2
        except Exception as e:
            pass
        
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
    
    def process_frame(self, img):
        self.frame_count += 1
        
        if self.frame_count % FACE_DETECTION_SKIP != 0:
            return img
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)
        
        current_faces = set()
        
        for res in results:
            if res['confidence'] < confidence_t:
                continue
            
            name, distance, pt_1, pt_2 = self.recognize_face(img_rgb, res)
            
            smoothed_name = self.smooth_recognition(name)
            
            if smoothed_name != 'unknown':
                current_faces.add(smoothed_name)
                attendance_marked = self.mark_attendance(smoothed_name)
                
                if attendance_marked:
                    self.detected_faces.add(smoothed_name)
            
            color = (0, 255, 0) if smoothed_name != 'unknown' else (0, 0, 255)
            cv2.rectangle(img, pt_1, pt_2, color, 2)
            
            label = f"{smoothed_name} ({1-distance:.2f})"
            cv2.putText(img, label, (pt_1[0], pt_1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img

    def start_recognition(self):
        if self.is_running:
            return

        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera.")
            self.is_running = False
            return

        self.start_time = time.time()
        self.detected_faces.clear()
        self.name_history.clear()
        self.update_frame()

    def update_frame(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                img_array = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.resize((self.camera_frame_label.winfo_width(), self.camera_frame_label.winfo_height()), Image.LANCZOS)
                self.current_image = ImageTk.PhotoImage(image=img_pil)
                self.camera_frame_label.config(image=self.current_image)
                self.camera_frame_label.image = self.current_image

                elapsed_time = time.time() - self.start_time
                if len(self.detected_faces) >= 1 or elapsed_time >= self.max_detection_time:
                    self.stop_recognition()
                    self.show_final_results()
                else:
                    self.root_window.after(10, self.update_frame)
            else:
                self.stop_recognition()
                messagebox.showerror("Camera Error", "Failed to read frame from camera.")
        else:
            self.stop_recognition()

    def stop_recognition(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        # Restore det1.jpg or a success message
        det_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\det1.jpg"))
        det_img_btn = det_img_btn.resize((self.camera_frame_label.winfo_width(), self.camera_frame_label.winfo_height()), Image.LANCZOS)
        self.current_image = ImageTk.PhotoImage(det_img_btn)
        self.camera_frame_label.config(image=self.current_image)
        self.camera_frame_label.image = self.current_image

    def show_final_results(self):
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

root=Tk()
# Open in full screen mode
root.attributes('-fullscreen', True)
root.title("Face_Recognition_System")

# Get screen dimensions for responsive design
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# This part is image labels setting start
# first header image - scale to screen width
img = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\banner.jpg"))
img = img.resize((screen_width, 130), Image.LANCZOS)  # Use Image.LANCZOS for resizing

photoimg = ImageTk.PhotoImage(img)

# set image as label
f_lb1 = Label(root, image=photoimg)
f_lb1.place(x=0, y=0, width=screen_width, height=130)

# background image - scale to screen dimensions
bg1 = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\bg3.jpg"))
bg1 = bg1.resize((screen_width, screen_height - 130), Image.LANCZOS)  # Use Image.LANCZOS for resizing
photobg1 = ImageTk.PhotoImage(bg1)

# set image as label
bg_img = Label(root, image=photobg1)
bg_img.place(x=0, y=130, width=screen_width, height=screen_height - 130)

# title section - responsive width
title_lb1 = Label(bg_img, text="Optimized Face Recognition Attendance System", font=("verdana", 28, "bold"),bg="white", fg="navyblue")
title_lb1.place(x=0, y=0, width=screen_width, height=45)

# ==================Function for Open Images Folder==================
def open_img():
    """Open the dataset folder"""
    dataset_path = os.path.join(os.getcwd(), "Dataset")
    if os.path.exists(dataset_path):
        os.startfile(dataset_path)
    else:
        messagebox.showinfo("Dataset Folder", "Dataset folder not found. Please create one first.")

# ==================Functions Buttons=====================
def student_pannels():
    """Open dataset folder for student management"""
    import subprocess
    subprocess.Popen(['python', 'generateDataset.py'], cwd=os.getcwd())

def train_pannels():
    """Train the optimized face recognition system"""
    import subprocess
    subprocess.Popen(['python', 'train_optimized.py'], cwd=os.getcwd())

def face_rec():
    """Start automated face recognition - opens camera, detects faces, marks attendance, and closes automatically"""
    global face_recognition_system
    if face_recognition_system is None:
        face_recognition_system = FaceRecognitionSystem(camera_frame, root)
    
    # Start recognition in a new thread to keep GUI responsive
    recognition_thread = threading.Thread(target=face_recognition_system.start_recognition)
    recognition_thread.daemon = True
    recognition_thread.start()

def attendance_pannel():
    """Open attendance viewer - shows today's attendance data immediately"""
    import subprocess
    import os
    
    try:
        # Use the quick attendance viewer for immediate results
        subprocess.Popen(['python', 'quick_attendance.py'], cwd=os.getcwd())
        print("Quick attendance viewer opened successfully!")
    except Exception as e:
        print(f"Error opening quick attendance viewer: {e}")
        # Fallback to simple viewer
        try:
            subprocess.Popen(['python', 'attendance_viewer.py'], cwd=os.getcwd())
            print("Attendance viewer opened successfully!")
        except Exception as e2:
            print(f"Error opening attendance viewer: {e2}")
            # Final fallback: show data directly in homepage
            show_attendance_data()

def developr():
    """Show system information"""
    messagebox.showinfo("System Info", "Face Recognition Attendance System\nVersion: 2.0 Optimized\n\nFeatures:\n• 100% Training Accuracy\n• Enhanced Recognition\n• Real-time Attendance\n• Confidence Smoothing")

def helpSupport():
    """Show help and support information"""
    messagebox.showinfo("Help & Support", "Face Recognition Attendance System Help\n\nSteps:\n1. Click 'Data Train' to train the system\n2. Click 'Face Detector' to start recognition\n3. Click 'Attendance' to view dashboard\n4. Click 'Dataset' to manage student images\n\nSystem Status: Optimized and Ready!")

def show_attendance_data():
    """Show attendance data in a simple window"""
    import csv
    from datetime import datetime
    import tkinter as tk
    from tkinter import ttk
    
    # Find today's attendance file
    today = datetime.now().strftime("%d-%m-%Y")
    attendance_file = f"att_{today}.csv"
    
    if not os.path.exists(attendance_file):
        messagebox.showinfo("Attendance", f"No attendance data found for today ({today}).\n\nMake sure to run face recognition first!")
        return
    
    # Create a simple attendance viewer
    viewer = tk.Toplevel()
    viewer.title(f"Attendance Data - {today}")
    viewer.geometry("600x400")
    viewer.configure(bg="white")
    
    # Title
    title_label = tk.Label(viewer, text=f"Attendance for {today}", 
                          font=("Arial", 16, "bold"), bg="white", fg="navyblue")
    title_label.pack(pady=10)
    
    # Create treeview for attendance data
    tree = ttk.Treeview(viewer, columns=("Date", "EntryTime", "ExitTime", "Status"), 
                       show="headings", height=15)
    
    # Define columns
    tree.heading("Date", text="Date")
    tree.heading("EntryTime", text="Entry Time")
    tree.heading("ExitTime", text="Exit Time")
    tree.heading("Status", text="Status")
    
    # Set column widths
    tree.column("Date", width=100)
    tree.column("EntryTime", width=100)
    tree.column("ExitTime", width=100)
    tree.column("Status", width=80)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(viewer, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack elements
    tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y")
    
    # Load attendance data
    try:
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:  # Ensure we have enough columns
                    tree.insert("", "end", values=row[:4])
    except Exception as e:
        messagebox.showerror("Error", f"Could not load attendance data: {e}")
    
    # Add close button
    close_btn = tk.Button(viewer, text="Close", command=viewer.destroy,
                         bg="#e74c3c", fg="white", font=("Arial", 12, "bold"))
    close_btn.pack(pady=10)

def Close():
    root.destroy()

def toggle_fullscreen(event=None):
    """Toggle between full screen and windowed mode with F11"""
    current_state = root.attributes('-fullscreen')
    root.attributes('-fullscreen', not current_state)

def exit_fullscreen(event=None):
    """Exit full screen mode with Escape key"""
    root.attributes('-fullscreen', False)

# Bind keyboard shortcuts
root.bind('<F11>', toggle_fullscreen)
root.bind('<Escape>', exit_fullscreen)

# Create buttons below the section
# -------------------------------------------------------------------------------------------------------------------
# Calculate responsive button positions based on screen width
button_width = 180
button_height = 180
text_height = 45
vertical_spacing = 100

# Calculate horizontal positions for 4 buttons across the screen
total_button_width = 4 * button_width  # 4 buttons
available_width = screen_width - total_button_width  # Remaining space
horizontal_spacing = available_width // 5  # Space between buttons and edges

# Row 1 positions (top row)
row1_y = 100
pos1_x = horizontal_spacing
pos2_x = pos1_x + button_width + horizontal_spacing
pos3_x = pos2_x + button_width + horizontal_spacing
pos4_x = pos3_x + button_width + horizontal_spacing

# Row 2 positions (bottom row)
row2_y = row1_y + button_height + text_height + 50

# student button 1
std_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\std1.jpg"))
std_img_btn = std_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
std_img1 = ImageTk.PhotoImage(std_img_btn)

std_b1 = Button(bg_img, command=student_pannels, image=std_img1, cursor="hand2")
std_b1.place(x=pos1_x, y=row1_y, width=button_width, height=button_height)

std_b1_1 = Button(bg_img, command=student_pannels, text="Dataset", cursor="hand2",
                  font=("tahoma", 15, "bold"), bg="white", fg="navyblue")
std_b1_1.place(x=pos1_x, y=row1_y + button_height + 10, width=button_width, height=text_height)

# Placeholder for camera feed or det1.jpg
camera_frame = Label(bg_img, bg="black")
camera_frame.place(x=pos2_x, y=row1_y, width=button_width, height=button_height)

# Load initial image for the camera frame
det_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\det1.jpg"))
det_img_btn = det_img_btn.resize((button_width, button_height), Image.LANCZOS)
det_img1 = ImageTk.PhotoImage(det_img_btn)
camera_frame.config(image=det_img1)
camera_frame.image = det_img1 # Keep a reference!

det_b1_1 = Button(bg_img, command=face_rec, text="Face Recognition", cursor="hand2", font=("tahoma", 15, "bold"),
                  bg="white", fg="navyblue")
det_b1_1.place(x=pos2_x, y=row1_y + button_height + 10, width=button_width, height=text_height)

# Attendance System button 3
att_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\att.jpg"))
att_img_btn = att_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
att_img1 = ImageTk.PhotoImage(att_img_btn)

att_b1 = Button(bg_img, command=attendance_pannel, image=att_img1, cursor="hand2")
att_b1.place(x=pos3_x, y=row1_y, width=button_width, height=button_height)

att_b1_1 = Button(bg_img, command=attendance_pannel, text="Attendance", cursor="hand2",
                  font=("tahoma", 15, "bold"), bg="white", fg="navyblue")
att_b1_1.place(x=pos3_x, y=row1_y + button_height + 10, width=button_width, height=text_height)

# Help Support button 4
hlp_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\hlp.jpg"))
hlp_img_btn = hlp_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
hlp_img1 = ImageTk.PhotoImage(hlp_img_btn)

hlp_b1 = Button(bg_img, command=helpSupport, image=hlp_img1, cursor="hand2")
hlp_b1.place(x=pos4_x, y=row1_y, width=button_width, height=button_height)

hlp_b1_1 = Button(bg_img, command=helpSupport, text="Help Support", cursor="hand2", font=("tahoma", 15, "bold"),
                  bg="white", fg="navyblue")
hlp_b1_1.place(x=pos4_x, y=row1_y + button_height + 10, width=button_width, height=text_height)

# Top 4 buttons end.......
# ---------------------------------------------------------------------------------------------------------------------------
# Start below buttons.........
# Train button 5
tra_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\tra1.jpg"))
tra_img_btn = tra_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
tra_img1 = ImageTk.PhotoImage(tra_img_btn)

tra_b1 = Button(bg_img, command=train_pannels, image=tra_img1, cursor="hand2")
tra_b1.place(x=pos1_x, y=row2_y, width=button_width, height=button_height)

tra_b1_1 = Button(bg_img, command=train_pannels, text="Train System", cursor="hand2", font=("tahoma", 15, "bold"),bg="white", fg="navyblue")
tra_b1_1.place(x=pos1_x, y=row2_y + button_height + 10, width=button_width, height=text_height)

# Photo button 6
pho_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\qr1.png"))
pho_img_btn = pho_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
pho_img1 = ImageTk.PhotoImage(pho_img_btn)

pho_b1 = Button(bg_img, command=open_img, image=pho_img1, cursor="hand2")
pho_b1.place(x=pos2_x, y=row2_y, width=button_width, height=button_height)

pho_b1_1 = Button(bg_img, command=open_img, text="View Dataset", cursor="hand2", font=("tahoma", 15, "bold"),
                  bg="white", fg="navyblue")
pho_b1_1.place(x=pos2_x, y=row2_y + button_height + 10, width=button_width, height=text_height)

# Developers button 7
dev_img_btn = Image.open(os.path.join(os.path.dirname(__file__), r"Images_GUI\dev.jpg"))
dev_img_btn = dev_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
dev_img1 = ImageTk.PhotoImage(dev_img_btn)

dev_b1 = Button(bg_img, command=developr, image=dev_img1, cursor="hand2")
dev_b1.place(x=pos3_x, y=row2_y, width=button_width, height=button_height)

dev_b1_1 = Button(bg_img, command=developr, text="Developers", cursor="hand2", font=("tahoma", 15, "bold"),
                  bg="white", fg="navyblue")
dev_b1_1.place(x=pos3_x, y=row2_y + button_height + 10, width=button_width, height=text_height)

# exit button 8
exi_img_btn = Image.open(r"Images_GUI\exi.jpg")
exi_img_btn = exi_img_btn.resize((button_width, button_height), Image.LANCZOS)  # Use Image.LANCZOS for resizing
exi_img1 = ImageTk.PhotoImage(exi_img_btn)

exi_b1 = Button(bg_img, command=Close, image=exi_img1, cursor="hand2")
exi_b1.place(x=pos4_x, y=row2_y, width=button_width, height=button_height)

exi_b1_1 = Button(bg_img, command=Close, text="Exit", cursor="hand2", font=("tahoma", 15, "bold"), bg="white",
                  fg="navyblue")
exi_b1_1.place(x=pos4_x, y=row2_y + button_height + 10, width=button_width, height=text_height)


root.mainloop()

