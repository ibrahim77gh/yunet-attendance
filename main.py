"""
YuNet Attendance System - Main Entry Point
A comprehensive face recognition-based attendance system.
"""

import cv2
import numpy as np
import os
from datetime import datetime

# Import from our modular structure
from utils import load_yunet_model, load_insightface_model, detect_faces_yunet, extract_face_embedding
from database import load_student_database
from attendance import process_attendance_for_image, export_attendance_to_excel

def main():
    """Main function to run attendance system with face recognition"""
    from config import SIMILARITY_THRESHOLD, CUTOFF_TIME
    
    print("Starting YuNet Attendance System...")
    
    # Load models
    print("Loading YuNet face detection model...")
    yunet_detector = load_yunet_model()
    
    print("Loading InsightFace buffalo_l model...")
    insightface_app = load_insightface_model()
    
    print("Loading student database...")
    students_db = load_student_database()
    
    if yunet_detector is None or insightface_app is None:
        print("Failed to load required models. Exiting.")
        return
    
    # Load sample image
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print("Creating a dummy image for demonstration...")
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = "dummy_sample.jpg"
        cv2.imwrite(image_path, dummy_image)
        print(f"Created dummy image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Process attendance for the image
    attendance_records = process_attendance_for_image(
        image, yunet_detector, insightface_app, students_db, 
        SIMILARITY_THRESHOLD, CUTOFF_TIME
    )
    
    # Export attendance records to Excel
    if attendance_records:
        print(f"\nExporting {len(attendance_records)} attendance records to Excel...")
        success = export_attendance_to_excel(attendance_records)
        
        if success:
            print("\n=== ATTENDANCE SUMMARY ===")
            for record in attendance_records:
                print(f"{record['Timestamp']}, {record['Student_ID']}, {record['Student_Name']}, {record['Classroom']}, {record['Status']}")
            
            print(f"\nProcessing completed successfully!")
            print(f"Created {len(attendance_records)} attendance records")
        else:
            print("Processing completed with export errors.")
    else:
        print("No attendance records created.")

if __name__ == "__main__":
    main()
