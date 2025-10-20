"""
Student database management functions.
"""

import os
import json
import numpy as np


def create_student_database():
    """Create a sample student database with embeddings"""
    from config import SAMPLE_STUDENTS, STUDENT_EMBEDDINGS_DIR, STUDENT_DATABASE_PATH
    
    students_db = {"students": SAMPLE_STUDENTS}
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(STUDENT_EMBEDDINGS_DIR, exist_ok=True)
    
    # Generate dummy embeddings for each student
    for student in students_db["students"]:
        embedding_path = os.path.join(STUDENT_EMBEDDINGS_DIR, student["embedding_file"])
        if not os.path.exists(embedding_path):
            # Generate a random 512-dimensional embedding
            dummy_embedding = np.random.rand(512).astype(np.float32)
            # Normalize the embedding
            dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)
            np.save(embedding_path, dummy_embedding)
            print(f"Created dummy embedding for {student['name']}: {embedding_path}")
    
    # Save student database
    with open(STUDENT_DATABASE_PATH, "w") as f:
        json.dump(students_db, f, indent=2)
    
    print("Student database created successfully!")
    return students_db


def load_student_database():
    """Load student database and embeddings"""
    from config import STUDENT_DATABASE_PATH, STUDENT_EMBEDDINGS_DIR
    
    if not os.path.exists(STUDENT_DATABASE_PATH):
        print("Student database not found. Creating sample database...")
        students_db = create_student_database()
    else:
        with open(STUDENT_DATABASE_PATH, "r") as f:
            students_db = json.load(f)
    
    # Load embeddings for each student
    for student in students_db["students"]:
        embedding_path = os.path.join(STUDENT_EMBEDDINGS_DIR, student["embedding_file"])
        if os.path.exists(embedding_path):
            student["embedding"] = np.load(embedding_path)
        else:
            print(f"Warning: Embedding not found for {student['name']}")
            student["embedding"] = None
    
    return students_db


def add_student_to_database(student_id, name, classroom, embedding):
    """Add a new student to the database"""
    # Load existing database
    if os.path.exists("student_database.json"):
        with open("student_database.json", "r") as f:
            students_db = json.load(f)
    else:
        students_db = {"students": []}
    
    # Create embedding file path
    embedding_filename = f"student_{student_id}_embedding.npy"
    embedding_path = os.path.join("student_embeddings", embedding_filename)
    
    # Ensure embeddings directory exists
    os.makedirs("student_embeddings", exist_ok=True)
    
    # Save embedding
    np.save(embedding_path, embedding)
    
    # Add student to database
    new_student = {
        "id": student_id,
        "name": name,
        "classroom": classroom,
        "embedding_file": embedding_filename
    }
    
    students_db["students"].append(new_student)
    
    # Save updated database
    with open("student_database.json", "w") as f:
        json.dump(students_db, f, indent=2)
    
    print(f"Added student {name} (ID: {student_id}) to database")
    return students_db
