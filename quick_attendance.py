#!/usr/bin/env python3
"""
Quick Attendance Viewer - Opens immediately with today's data
Simple and fast attendance viewing without complex UI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
from datetime import datetime

def show_quick_attendance():
    """Show today's attendance data in a simple window"""
    # Create the window
    window = tk.Tk()
    window.title("Quick Attendance Viewer")
    window.geometry("500x400")
    window.configure(bg="#f8f9fa")
    window.resizable(True, True)
    
    # Title
    today = datetime.now().strftime("%B %d, %Y")
    title_label = tk.Label(window, text=f"Attendance - {today}", 
                          font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
    title_label.pack(fill=tk.X, pady=(0, 10))
    
    # Find today's attendance file
    today_file = f"att_{datetime.now().strftime('%d-%m-%Y')}.csv"
    
    if not os.path.exists(today_file):
        # No attendance file, show message
        no_data_label = tk.Label(window, 
                                text="No attendance recorded today!\n\nClick 'Start Recognition' first to mark attendance.",
                                font=("Arial", 12), bg="#f8f9fa", fg="#7f8c8d",
                                justify=tk.CENTER, wraplength=400)
        no_data_label.pack(pady=50)
        
        # Instructions
        instructions = tk.Label(window, 
                               text="Steps:\n1. Click 'Face Recognition' to start\n2. Let system recognize faces\n3. Click 'Attendance' again to view results",
                               font=("Arial", 10), bg="#f8f9fa", fg="#95a5a6",
                               justify=tk.CENTER)
        instructions.pack(pady=20)
    else:
        # Create frame for data
        data_frame = tk.Frame(window, bg="#f8f9fa")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Headers
        headers_frame = tk.Frame(data_frame, bg="#ecf0f1")
        headers_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(headers_frame, text="Name", font=("Arial", 11, "bold"), 
                bg="#ecf0f1", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Label(headers_frame, text="Entry Time", font=("Arial", 11, "bold"), 
                bg="#ecf0f1", width=12, anchor=tk.CENTER).pack(side=tk.LEFT, padx=5)
        tk.Label(headers_frame, text="Status", font=("Arial", 11, "bold"), 
                bg="#ecf0f1", width=10, anchor=tk.CENTER).pack(side=tk.LEFT, padx=5)
        
        # Scrollable area for data
        scroll_frame = tk.Frame(data_frame)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(scroll_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Load and display data
        total_records = 0
        try:
            with open(today_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 4:  # Ensure we have enough data
                        name, date, entry_time, exit_time, status = row[:5]
                        
                        # Create row frame
                        row_frame = tk.Frame(scrollable_frame, bg="#ffffff", relief=tk.RAISED, bd=1)
                        row_frame.pack(fill=tk.X, pady=2, padx=5)
                        
                        # Name
                        tk.Label(row_frame, text=name, font=("Arial", 10), 
                                bg="#ffffff", width=20, anchor=tk.W).pack(side=tk.LEFT, padx=5)
                        
                        # Entry Time
                        tk.Label(row_frame, text=entry_time, font=("Arial", 10), 
                                bg="#ffffff", width=15, anchor=tk.CENTER).pack(side=tk.LEFT, padx=5)
                        
                        # Status with color
                        status_label = tk.Label(row_frame, text=status, font=("Arial", 10, "bold"), 
                                               bg="#ffffff", width=10, anchor=tk.CENTER)
                        status_label.pack(side=tk.LEFT, padx=5)
                        
                        # Color code status
                        if status.lower() == "present":
                            status_label.config(fg="#27ae60")  # Green
                        else:
                            status_label.config(fg="#e74c3c")  # Red
                        
                        total_records += 1
        
        except Exception as e:
            error_label = tk.Label(scrollable_frame, 
                                 text=f"Error loading data: {e}",
                                 font=("Arial", 10), fg="#e74c3c", bg="#f8f9fa")
            error_label.pack(pady=20)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Summary
        summary_label = tk.Label(data_frame, 
                               text=f"Total Records: {total_records}",
                               font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
        summary_label.pack(pady=(10, 5))
    
    # Close button
    close_btn = tk.Button(window, text="Close", command=window.destroy,
                         bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), 
                         width=10, height=2)
    close_btn.pack(pady=15)
    
    # Make window appear in center
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')
    
    return window

if __name__ == "__main__":
    window = show_quick_attendance()
    window.mainloop()