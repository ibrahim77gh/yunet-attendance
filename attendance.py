"""
Attendance system logic including face matching and status determination.
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from sklearn.metrics.pairwise import cosine_similarity


def match_face_to_student(detected_embedding, students_db, similarity_threshold=0.6):
    """Match detected face embedding to a student in the database"""
    if detected_embedding is None:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    for student in students_db["students"]:
        if student["embedding"] is not None:
            # Calculate cosine similarity
            similarity = cosine_similarity(
                detected_embedding.reshape(1, -1),
                student["embedding"].reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = student
    
    return best_match, best_similarity


def determine_attendance_status(detection_time, cutoff_time_str="08:00"):
    """Determine attendance status based on detection time and cutoff time"""
    try:
        # Parse cutoff time
        cutoff_hour, cutoff_minute = map(int, cutoff_time_str.split(":"))
        cutoff_time = time(cutoff_hour, cutoff_minute)
        
        # Get detection time
        detection_time_only = detection_time.time()
        
        # Determine status
        if detection_time_only <= cutoff_time:
            return "Present"
        else:
            return "Late"
            
    except Exception as e:
        print(f"Error determining attendance status: {e}")
        return "Unknown"


def create_attendance_record(detection_time, student_info, similarity_score, status, cutoff_time="08:00"):
    """Create an attendance record for Excel export"""
    timestamp_str = detection_time.strftime("%Y-%m-%d %H:%M")
    
    record = {
        'Timestamp': timestamp_str,
        'Student_ID': student_info['id'] if student_info else 'Unknown',
        'Student_Name': student_info['name'] if student_info else 'Unknown',
        'Classroom': student_info['classroom'] if student_info else 'Unknown',
        'Status': status,
        'Similarity_Score': round(similarity_score, 4) if similarity_score > 0 else 0.0,
        'Cutoff_Time': cutoff_time,
        'Detection_Date': detection_time.date(),
        'Detection_Time': detection_time.time()
    }
    
    return record


def export_attendance_to_excel(attendance_records, filename="attendance_report.xlsx"):
    """Export attendance records to Excel file"""
    try:
        if not attendance_records:
            print("No attendance records to export")
            return False
            
        df = pd.DataFrame(attendance_records)
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main attendance sheet
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Summary sheet
            summary_data = {
                'Total_Detections': [len(attendance_records)],
                'Present_Count': [len([r for r in attendance_records if r['Status'] == 'Present'])],
                'Late_Count': [len([r for r in attendance_records if r['Status'] == 'Late'])],
                'Unknown_Count': [len([r for r in attendance_records if r['Status'] == 'Unknown'])],
                'Report_Generated': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Attendance report exported to {filename}")
        print(f"Total records: {len(attendance_records)}")
        return True
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False


def process_attendance_for_image(image, yunet_detector, insightface_app, students_db, 
                                similarity_threshold=0.6, cutoff_time="08:00"):
    """Process attendance for a single image"""
    from utils import detect_faces_yunet, extract_face_embedding
    
    # Detect faces using YuNet
    faces = detect_faces_yunet(yunet_detector, image)
    
    # Process each detected face
    attendance_records = []
    detection_time = datetime.now()
    
    for i, face in enumerate(faces):
        print(f"\nProcessing face {i+1}...")
        
        # Extract embedding
        embedding = extract_face_embedding(insightface_app, image, face)
        
        if embedding is not None:
            print(f"Face {i+1}: Successfully extracted embedding (shape: {embedding.shape})")
            
            # Match face to student
            matched_student, similarity_score = match_face_to_student(
                embedding, students_db, similarity_threshold
            )
            
            if matched_student:
                print(f"Face {i+1}: Matched to {matched_student['name']} (ID: {matched_student['id']})")
                print(f"Similarity score: {similarity_score:.4f}")
                
                # Determine attendance status
                status = determine_attendance_status(detection_time, cutoff_time)
                print(f"Attendance status: {status}")
                
                # Create attendance record
                record = create_attendance_record(
                    detection_time, matched_student, similarity_score, status, cutoff_time
                )
                attendance_records.append(record)
                
            else:
                print(f"Face {i+1}: No matching student found (similarity < {similarity_threshold})")
                
                # Create record for unknown person
                unknown_student = {
                    'id': 'Unknown',
                    'name': 'Unknown Person',
                    'classroom': 'Unknown'
                }
                record = create_attendance_record(
                    detection_time, unknown_student, 0.0, "Unknown", cutoff_time
                )
                attendance_records.append(record)
                
        else:
            print(f"Face {i+1}: Failed to extract embedding")
    
    return attendance_records
