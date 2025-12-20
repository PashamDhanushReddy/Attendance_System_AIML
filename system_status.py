#!/usr/bin/env python3
"""
System Status Checker for Face Recognition Attendance System
Shows current status of all modules and system components
"""

import os
import sys
import json
from datetime import datetime

def check_system_status():
    """Check the status of all system components"""
    print("=" * 60)
    print("🎯 FACE RECOGNITION ATTENDANCE SYSTEM - STATUS CHECK")
    print("=" * 60)
    
    # Check working directory
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check available modules
    modules = {
        'homepage.py': 'Main GUI Interface',
        'train_optimized.py': 'Optimized Training System',
        'recognizer_optimized.py': 'Enhanced Face Recognition',
        'dashboard.py': 'Professional Dashboard',
        'generateDataset.py': 'Dataset Management',
        'architecture.py': 'FaceNet Model Architecture',
        'facenet_keras_weights.h5': 'Pre-trained Weights'
    }
    
    print("\n📋 AVAILABLE MODULES:")
    print("-" * 40)
    
    all_modules_exist = True
    for module, description in modules.items():
        if os.path.exists(module):
            print(f"✅ {module:<25} - {description}")
        else:
            print(f"❌ {module:<25} - {description}")
            all_modules_exist = False
    
    # Check models and encodings
    print("\n🧠 MODELS & ENCODINGS:")
    print("-" * 40)
    
    model_files = {
        'classifier/classifier_optimized.pkl': 'Optimized Classifier',
        'encodings/encodings_optimized.pkl': 'Optimized Encodings',
        'encodings/training_stats.json': 'Training Statistics'
    }
    
    models_exist = True
    for model_file, description in model_files.items():
        if os.path.exists(model_file):
            print(f"✅ {model_file:<35} - {description}")
            
            # Show training stats if available
            if model_file == 'encodings/training_stats.json':
                try:
                    with open(model_file, 'r') as f:
                        stats = json.load(f)
                    print(f"   📊 Best Classifier: {stats.get('best_classifier', 'Unknown')}")
                    print(f"   🎯 Accuracy: {stats.get('best_accuracy', 0) * 100:.1f}%")
                    print(f"   👥 People: {stats.get('total_people', 0)}")
                    print(f"   📸 Encodings: {stats.get('total_encodings', 0)}")
                except:
                    pass
        else:
            print(f"❌ {model_file:<35} - {description}")
            models_exist = False
    
    # Check datasets
    print("\n📸 DATASETS:")
    print("-" * 40)
    
    datasets = ['Dataset', 'Dataset_Enhanced']
    for dataset in datasets:
        if os.path.exists(dataset):
            people = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
            print(f"✅ {dataset:<20} - {len(people)} people found")
            for person in people:
                person_path = os.path.join(dataset, person)
                images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"   👤 {person:<15} - {len(images)} images")
        else:
            print(f"❌ {dataset:<20} - Not found")
    
    # Check attendance files
    print("\n📊 ATTENDANCE FILES:")
    print("-" * 40)
    
    attendance_files = [f for f in os.listdir('.') if f.startswith('att_') and f.endswith('.csv')]
    if attendance_files:
        for file in attendance_files:
            print(f"✅ {file}")
    else:
        print("ℹ️  No attendance files found yet")
    
    # Overall system status
    print("\n" + "=" * 60)
    if all_modules_exist and models_exist:
        print("🎉 SYSTEM STATUS: FULLY OPERATIONAL")
        print("   Your optimized face recognition system is ready to use!")
        print("   Use 'python homepage.py' to start the main interface.")
    else:
        print("⚠️  SYSTEM STATUS: PARTIALLY READY")
        if not all_modules_exist:
            print("   Missing core modules - please check the installation.")
        if not models_exist:
            print("   Missing trained models - please run training first.")
    
    print("=" * 60)

if __name__ == "__main__":
    check_system_status()