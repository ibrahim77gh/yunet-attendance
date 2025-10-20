#!/usr/bin/env python3
"""
Test script to demonstrate the attendance system with a simulated successful match.
This script modifies one of the student embeddings to match the detected face.
"""

import numpy as np
import os
import cv2
from datetime import datetime

# Import from our modular structure
from utils import load_yunet_model, load_insightface_model, detect_faces_yunet, extract_face_embedding
from database import load_student_database
from attendance import match_face_to_student, create_attendance_record, export_attendance_to_excel, determine_attendance_status

def simulate_successful_match():
    """Simulate a successful face match by modifying a student embedding"""
    print("=== DEMO: Simulating Successful Face Match ===")
    
    # Load the student database
    students_db = load_student_database()
    
    # Load models
    yunet_detector = load_yunet_model()
    insightface_app = load_insightface_model()
    
    if yunet_detector is None or insightface_app is None:
        print("Failed to load models")
        return
    
    # Load sample image
    image_path = "sample_image.jpg"
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Detect faces
    faces = detect_faces_yunet(yunet_detector, image)
    if len(faces) == 0:
        print("No faces detected")
        return
    
    # Extract embedding from first face
    embedding = extract_face_embedding(insightface_app, image, faces[0])
    if embedding is None:
        print("Failed to extract embedding")
        return
    
    print(f"Extracted embedding shape: {embedding.shape}")
    
    # Modify Ahmed Ali's embedding to be similar to the detected face
    # This simulates a successful match
    ahmed_embedding_path = "student_embeddings/student_101_embedding.npy"
    if os.path.exists(ahmed_embedding_path):
        # Create a modified embedding that will match
        modified_embedding = embedding + np.random.normal(0, 0.1, embedding.shape)
        modified_embedding = modified_embedding / np.linalg.norm(modified_embedding)
        np.save(ahmed_embedding_path, modified_embedding)
        print("Modified Ahmed Ali's embedding to simulate a match")
    
    # Reload the student database with the modified embedding
    students_db = load_student_database()
    
    # Now test the matching
    matched_student, similarity_score = match_face_to_student(embedding, students_db, 0.6)
    
    if matched_student:
        print(f"\nSUCCESSFUL MATCH!")
        print(f"Matched to: {matched_student['name']} (ID: {matched_student['id']})")
        print(f"Classroom: {matched_student['classroom']}")
        print(f"Similarity score: {similarity_score:.4f}")
        
        # Determine attendance status
        detection_time = datetime.now()
        status = determine_attendance_status(detection_time, "08:00")
        print(f"Attendance status: {status}")
        
        # Create attendance record
        record = create_attendance_record(
            detection_time, matched_student, similarity_score, status, "08:00"
        )
        
        # Export to Excel
        success = export_attendance_to_excel([record], "demo_attendance_report.xlsx")
        
        if success:
            print(f"\nDemo attendance record:")
            print(f"{record['Timestamp']}, {record['Student_ID']}, {record['Student_Name']}, {record['Classroom']}, {record['Status']}")
            print(f"\nDemo completed successfully! Check 'demo_attendance_report.xlsx'")
        
    else:
        print("No match found")

if __name__ == "__main__":
    simulate_successful_match()
