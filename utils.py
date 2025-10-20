"""
Utility functions for model loading and basic operations.
"""

import cv2
import numpy as np
import os
from insightface import app


def load_yunet_model(model_path=None):
    """Load YuNet face detection model"""
    from config import YUNET_MODEL_PATH, YUNET_INPUT_SIZE, YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K
    
    if model_path is None:
        model_path = YUNET_MODEL_PATH
        
    if not os.path.exists(model_path):
        print(f"Warning: YuNet model not found at {model_path}")
        print("Please download face_detection_yunet_2023mar.onnx from OpenCV Zoo")
        return None
    
    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        YUNET_INPUT_SIZE,
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K
    )
    return detector


def load_insightface_model():
    """Load InsightFace buffalo_l model for face recognition"""
    from config import INSIGHTFACE_MODEL_NAME, INSIGHTFACE_DET_SIZE, INSIGHTFACE_PROVIDERS
    
    try:
        # Initialize InsightFace app
        face_app = app.FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=INSIGHTFACE_PROVIDERS)
        face_app.prepare(ctx_id=0, det_size=INSIGHTFACE_DET_SIZE)
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
