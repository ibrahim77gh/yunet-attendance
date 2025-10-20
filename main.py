import cv2
import numpy as np
import pandas as pd
from insightface import app
from insightface.model_zoo import get_model
import os
from datetime import datetime

def load_yunet_model(model_path="face_detection_yunet_2023mar.onnx"):
    """Load YuNet face detection model"""
    if not os.path.exists(model_path):
        print(f"Warning: YuNet model not found at {model_path}")
        print("Please download face_detection_yunet_2023mar.onnx from OpenCV Zoo")
        return None
    
    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),  # input size
        0.6,  # score threshold
        0.3,  # nms threshold
        5000  # top k
    )
    return detector

def load_insightface_model():
    """Load InsightFace buffalo_l model for face recognition"""
    try:
        # Initialize InsightFace app
        face_app = app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        return face_app
    except Exception as e:
        print(f"Error loading InsightFace model: {e}")
        print("Make sure insightface is installed: pip install insightface")
        return None

def detect_faces_yunet(detector, image):
    """Detect faces using YuNet"""
    if detector is None:
        return []
    
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    
    _, faces = detector.detect(image)
    if faces is None:
        return []
    
    return faces

def extract_face_embedding(face_app, image, face_box):
    """Extract face embedding using InsightFace"""
    if face_app is None:
        return None
    
    try:
        # Instead of cropping, let InsightFace detect faces in the entire image
        faces = face_app.get(image)
        
        if len(faces) == 0:
            return None
        
        # Convert YuNet face box to format for matching
        yunet_x, yunet_y, yunet_w, yunet_h = face_box[:4].astype(int)
        yunet_center_x = yunet_x + yunet_w // 2
        yunet_center_y = yunet_y + yunet_h // 2
        
        # Find the InsightFace detection that best matches the YuNet detection
        best_match = None
        min_distance = float('inf')
        
        for face in faces:
            # Get InsightFace bounding box
            iface_bbox = face.bbox
            iface_x, iface_y, iface_w, iface_h = iface_bbox
            iface_center_x = iface_x + iface_w // 2
            iface_center_y = iface_y + iface_h // 2
            
            # Calculate distance between centers
            distance = ((yunet_center_x - iface_center_x) ** 2 + 
                       (yunet_center_y - iface_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = face
        
        if best_match is not None:
            embedding = best_match.embedding
            return embedding
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def create_sample_record(image_path, face_count, embeddings):
    """Create a sample record for Excel export"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    record = {
        'timestamp': timestamp,
        'image_path': image_path,
        'face_count': face_count,
        'processing_date': datetime.now().date(),
        'processing_time': datetime.now().time()
    }
    
    # Add embedding data (first 5 dimensions for readability)
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            record[f'face_{i+1}_embedding_dim_1'] = float(embedding[0])
            record[f'face_{i+1}_embedding_dim_2'] = float(embedding[1])
            record[f'face_{i+1}_embedding_dim_3'] = float(embedding[2])
            record[f'face_{i+1}_embedding_dim_4'] = float(embedding[3])
            record[f'face_{i+1}_embedding_dim_5'] = float(embedding[4])
            record[f'face_{i+1}_embedding_norm'] = float(np.linalg.norm(embedding))
        else:
            record[f'face_{i+1}_embedding_dim_1'] = None
            record[f'face_{i+1}_embedding_dim_2'] = None
            record[f'face_{i+1}_embedding_dim_3'] = None
            record[f'face_{i+1}_embedding_dim_4'] = None
            record[f'face_{i+1}_embedding_dim_5'] = None
            record[f'face_{i+1}_embedding_norm'] = None
    
    return record

def export_to_excel(record, filename="face_analysis_results.xlsx"):
    """Export record to Excel file"""
    try:
        df = pd.DataFrame([record])
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Results exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False

def main():
    """Main function to run face detection and embedding extraction"""
    print("Starting face detection and embedding extraction...")
    
    # Load models
    print("Loading YuNet face detection model...")
    yunet_detector = load_yunet_model()
    
    print("Loading InsightFace buffalo_l model...")
    insightface_app = load_insightface_model()
    
    if yunet_detector is None or insightface_app is None:
        print("Failed to load required models. Exiting.")
        return
    
    # Load sample image (you can replace this with your image path)
    image_path = "sample_image.jpg"  # Replace with your image path
    
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print("Please provide a valid image path or place a sample_image.jpg in the current directory")
        
        # Create a dummy image for demonstration
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
    
    # Detect faces using YuNet
    print("Detecting faces with YuNet...")
    faces = detect_faces_yunet(yunet_detector, image)
    print(f"Found {len(faces)} faces")
    
    # Extract embeddings for each face
    embeddings = []
    for i, face in enumerate(faces):
        embedding = extract_face_embedding(insightface_app, image, face)
        embeddings.append(embedding)
        
        if embedding is not None:
            print(f"Face {i+1}: Successfully extracted embedding (shape: {embedding.shape})")
        else:
            print(f"Face {i+1}: Failed to extract embedding")
    
    # Create sample record
    print("Creating sample record...")
    record = create_sample_record(image_path, len(faces), embeddings)
    
    # Export to Excel
    print("Exporting to Excel...")
    success = export_to_excel(record)
    
    if success:
        print("Processing completed successfully!")
        print(f"Found {len(faces)} faces")
        print(f"Successfully extracted {len([e for e in embeddings if e is not None])} embeddings")
    else:
        print("Processing completed with errors.")

if __name__ == "__main__":
    main()
