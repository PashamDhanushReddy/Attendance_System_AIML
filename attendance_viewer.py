#!/usr/bin/env python3
"""
Simple Attendance Viewer
Shows attendance data in a user-friendly window
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
from datetime import datetime

class AttendanceViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Attendance Viewer")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.setup_ui()
        self.load_attendance_data()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="Attendance Records", 
                              font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=15)
        
        # Main content
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Controls
        control_frame = tk.Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date selection
        tk.Label(control_frame, text="Select Date:", font=("Arial", 12), bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 10))
        
        self.date_var = tk.StringVar()
        self.date_combo = ttk.Combobox(control_frame, textvariable=self.date_var, width=15, font=("Arial", 11))
        self.date_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.date_combo.bind("<<ComboboxSelected>>", self.on_date_change)
        
        # Refresh button
        refresh_btn = tk.Button(control_frame, text="Refresh", command=self.load_attendance_data,
                               bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=10)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Export button
        export_btn = tk.Button(control_frame, text="Export to Excel", command=self.export_data,
                              bg="#3498db", fg="white", font=("Arial", 11, "bold"), width=12)
        export_btn.pack(side=tk.LEFT)
        
        # Attendance data frame
        data_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for attendance data
        self.tree = ttk.Treeview(data_frame, columns=("Date", "EntryTime", "ExitTime", "Status"), 
                                 show="headings", height=20)
        
        # Define columns
        self.tree.heading("Date", text="Date")
        self.tree.heading("EntryTime", text="Entry Time")
        self.tree.heading("ExitTime", text="Exit Time")
        self.tree.heading("Status", text="Status")
        
        # Set column widths
        self.tree.column("Date", width=120)
        self.tree.column("EntryTime", width=120)
        self.tree.column("ExitTime", width=120)
        self.tree.column("Status", width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack elements
        self.tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_attendance_data(self):
        """Load available attendance files and data"""
        # Find all attendance files
        attendance_files = [f for f in os.listdir('.') if f.startswith('att_') and f.endswith('.csv')]
        
        if not attendance_files:
            self.status_bar.config(text="No attendance files found")
            return
        
        # Extract dates from filenames
        dates = []
        for file in attendance_files:
            try:
                date_str = file.replace('att_', '').replace('.csv', '')
                # Try to parse the date
                date_obj = datetime.strptime(date_str, "%d-%m-%Y")
                dates.append(date_str)
            except:
                continue
        
        dates.sort(reverse=True)  # Most recent first
        
        # Update combobox
        self.date_combo['values'] = dates
        if dates:
            self.date_var.set(dates[0])  # Select most recent
            self.load_specific_date(dates[0])
    
    def on_date_change(self, event=None):
        """Handle date selection change"""
        selected_date = self.date_var.get()
        if selected_date:
            self.load_specific_date(selected_date)
    
    def load_specific_date(self, date_str):
        """Load attendance data for specific date"""
        attendance_file = f"att_{date_str}.csv"
        
        if not os.path.exists(attendance_file):
            self.status_bar.config(text=f"File {attendance_file} not found")
            return
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            with open(attendance_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                row_count = 0
                for row in reader:
                    if len(row) >= 4:  # Ensure we have enough columns
                        self.tree.insert("", "end", values=row[:4])
                        row_count += 1
                
                self.status_bar.config(text=f"Loaded {row_count} attendance records for {date_str}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load attendance data: {e}")
            self.status_bar.config(text="Error loading data")
    
    def export_data(self):
        """Export current data to Excel format"""
        try:
            import pandas as pd
            
            # Get current data from treeview
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                if values:
                    data.append(values)
            
            if not data:
                messagebox.showwarning("Export", "No data to export")
                return
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=["Date", "EntryTime", "ExitTime", "Status"])
            
            # Save to Excel
            filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            
            messagebox.showinfo("Export", f"Data exported to {filename}")
            self.status_bar.config(text=f"Data exported to {filename}")
            
        except ImportError:
            messagebox.showwarning("Export", "pandas library not found. Install with: pip install pandas openpyxl")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export data: {e}")
    
    def run(self):
        """Run the attendance viewer"""
        self.root.mainloop()

def main():
    """Main function"""
    viewer = AttendanceViewer()
    viewer.run()

if __name__ == "__main__":
    main()