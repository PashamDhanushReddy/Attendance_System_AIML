import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from datetime import datetime
import csv
import json

class AttendanceDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System - Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # System state
        self.is_running = False
        self.camera = None
        self.current_frame = None
        self.attendance_data = {}
        self.total_recognitions = 0
        
        self.setup_ui()
        self.load_attendance_data()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="Face Recognition Attendance System", 
                              font=("Arial", 24, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=20)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Camera and controls
        left_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera feed
        self.camera_label = tk.Label(left_frame, bg="black", width=640, height=480)
        self.camera_label.pack(pady=10)
        
        # Control buttons
        control_frame = tk.Frame(left_frame, bg="white")
        control_frame.pack(pady=10)
        
        self.start_button = tk.Button(control_frame, text="Start Recognition", 
                                     command=self.start_recognition,
                                     bg="#27ae60", fg="white", font=("Arial", 12, "bold"),
                                     width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="Stop Recognition", 
                                    command=self.stop_recognition,
                                    bg="#e74c3c", fg="white", font=("Arial", 12, "bold"),
                                    width=15, height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Statistics and attendance
        right_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Statistics section
        stats_frame = tk.LabelFrame(right_frame, text="System Statistics", 
                                   font=("Arial", 14, "bold"), bg="white")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_labels = {}
        stats_items = [
            ("Total Recognitions", "0"),
            ("People in Dataset", "0"),
            ("Today's Attendance", "0"),
            ("System Status", "Stopped")
        ]
        
        for i, (label, default) in enumerate(stats_items):
            tk.Label(stats_frame, text=f"{label}:", font=("Arial", 11), 
                    bg="white", anchor=tk.W).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            self.stats_labels[label] = tk.Label(stats_frame, text=default, 
                                              font=("Arial", 11, "bold"), bg="white")
            self.stats_labels[label].grid(row=i, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Attendance section
        attendance_frame = tk.LabelFrame(right_frame, text="Today's Attendance", 
                                        font=("Arial", 14, "bold"), bg="white")
        attendance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview for attendance data
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=("Time", "Status"), 
                                           show="tree headings", height=15)
        self.attendance_tree.heading("#0", text="Name")
        self.attendance_tree.heading("Time", text="Entry Time")
        self.attendance_tree.heading("Status", text="Status")
        
        self.attendance_tree.column("#0", width=150)
        self.attendance_tree.column("Time", width=100)
        self.attendance_tree.column("Status", width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(attendance_frame, orient=tk.VERTICAL, 
                                 command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom section - Recent activity
        bottom_frame = tk.Frame(self.root, bg="#34495e", height=100)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)
        
        self.activity_label = tk.Label(bottom_frame, text="System ready. Click 'Start Recognition' to begin.",
                                      font=("Arial", 11), fg="white", bg="#34495e")
        self.activity_label.pack(pady=20)
        
        # Load dataset info
        self.update_dataset_info()
        
    def update_dataset_info(self):
        """Update dataset information"""
        try:
            dataset_path = "Dataset"
            if os.path.exists(dataset_path):
                people_count = len([d for d in os.listdir(dataset_path) 
                                  if os.path.isdir(os.path.join(dataset_path, d))])
                self.stats_labels["People in Dataset"].config(text=str(people_count))
        except Exception as e:
            print(f"Error updating dataset info: {e}")
    
    def load_attendance_data(self):
        """Load today's attendance data"""
        try:
            today = datetime.now().strftime("%d-%m-%Y")
            attendance_file = f"att_{today}.csv"
            
            if os.path.exists(attendance_file):
                with open(attendance_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    
                    for row in reader:
                        if len(row) >= 3:
                            name, date, entry_time = row[0], row[1], row[2]
                            self.attendance_data[name] = {
                                'entry_time': entry_time,
                                'status': 'Present'
                            }
                            
                            # Add to treeview
                            self.attendance_tree.insert("", tk.END, text=name,
                                                       values=(entry_time, 'Present'))
                
                self.stats_labels["Today's Attendance"].config(text=str(len(self.attendance_data)))
                
        except Exception as e:
            print(f"Error loading attendance data: {e}")
    
    def start_recognition(self):
        """Start face recognition"""
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stats_labels["System Status"].config(text="Running", fg="green")
        self.activity_label.config(text="Face recognition started. Looking for faces...")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def stop_recognition(self):
        """Stop face recognition"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.stats_labels["System Status"].config(text="Stopped", fg="red")
        self.activity_label.config(text="Face recognition stopped.")
        
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def camera_loop(self):
        """Camera processing loop"""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            self.activity_label.config(text="Error: Cannot open camera")
            return
        
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                self.activity_label.config(text="Error: Cannot read frame")
                break
            
            # Here you would integrate your actual face recognition
            # For now, just display the camera feed
            self.current_frame = frame
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update camera display
            self.camera_label.config(image=frame_tk)
            self.camera_label.image = frame_tk
            
            time.sleep(0.03)  # ~30 FPS
        
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def add_attendance_entry(self, name):
        """Add new attendance entry"""
        current_time = datetime.now().strftime("%I:%M %p")
        
        if name not in self.attendance_data:
            self.attendance_data[name] = {
                'entry_time': current_time,
                'status': 'Present'
            }
            
            # Add to treeview
            self.attendance_tree.insert("", tk.END, text=name,
                                       values=(current_time, 'Present'))
            
            # Update statistics
            self.stats_labels["Today's Attendance"].config(text=str(len(self.attendance_data)))
            self.total_recognitions += 1
            self.stats_labels["Total Recognitions"].config(text=str(self.total_recognitions))
            
            # Update activity
            self.activity_label.config(text=f"Recognized: {name} at {current_time}")
    
    def run(self):
        """Run the dashboard"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_recognition()
            self.root.quit()
        finally:
            if self.camera:
                self.camera.release()

def main():
    """Main function"""
    dashboard = AttendanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()