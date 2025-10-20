"""
Configuration settings for the YuNet Attendance System.
"""

# Face matching configuration
SIMILARITY_THRESHOLD = 0.6  # Threshold for face matching (0.0-1.0)
CUTOFF_TIME = "08:00"       # Attendance cutoff time (HH:MM format)

# Model configuration
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

# InsightFace configuration
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]

# File paths
STUDENT_DATABASE_PATH = "student_database.json"
STUDENT_EMBEDDINGS_DIR = "student_embeddings"
DEFAULT_OUTPUT_FILE = "attendance_report.xlsx"

# Sample students configuration
SAMPLE_STUDENTS = [
    {
        "id": "101",
        "name": "Ahmed Ali",
        "classroom": "Grade 1-A",
        "embedding_file": "student_101_embedding.npy"
    },
    {
        "id": "102", 
        "name": "Sarah Johnson",
        "classroom": "Grade 1-A",
        "embedding_file": "student_102_embedding.npy"
    },
    {
        "id": "103",
        "name": "Mohammed Hassan",
        "classroom": "Grade 1-B", 
        "embedding_file": "student_103_embedding.npy"
    }
]
