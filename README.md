# YuNet Attendance System

A comprehensive face recognition-based attendance system using YuNet for face detection and InsightFace for face recognition, with automatic Excel reporting.

## Features

- **Face Detection**: Uses YuNet ONNX model for accurate face detection
- **Face Recognition**: Leverages InsightFace buffalo_l model for face embedding extraction
- **Student Database**: Manages student information with precomputed face embeddings (.npy files)
- **Attendance Classification**: Automatically determines Present/Late/Absent status based on configurable cutoff time
- **Excel Reporting**: Generates comprehensive attendance reports with multiple sheets
- **Configurable Parameters**: Adjustable similarity threshold and cutoff time

## System Requirements

- Python 3.8+
- Windows/Linux/macOS
- Virtual environment recommended

## Installation

1. **Clone or download the project files**

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YuNet model**:
   - Download `face_detection_yunet_2023mar.onnx` from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
   - Place it in the project root directory

## Usage

### Basic Usage

Run the main attendance system:
```bash
python main.py
```

### Demo with Successful Match

Run the test script that simulates a successful face match:
```bash
python test_attendance_system.py
```

## Configuration

### Centralized Settings

All configuration is managed through `config.py`:

```python
# Face matching configuration
SIMILARITY_THRESHOLD = 0.6  # Threshold for face matching (0.0-1.0)
CUTOFF_TIME = "08:00"       # Attendance cutoff time (HH:MM format)

# Model configuration
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
INSIGHTFACE_MODEL_NAME = "buffalo_l"

# File paths
STUDENT_DATABASE_PATH = "student_database.json"
STUDENT_EMBEDDINGS_DIR = "student_embeddings"
```

### Student Database

The system automatically creates a sample student database (`student_database.json`) with three students:

- **Ahmed Ali** (ID: 101, Grade 1-A)
- **Sarah Johnson** (ID: 102, Grade 1-A)  
- **Mohammed Hassan** (ID: 103, Grade 1-B)

Each student has a corresponding face embedding file in the `student_embeddings/` directory.


## Output Files

### Excel Reports

The system generates Excel files with two sheets:

1. **Attendance Sheet**: Contains individual attendance records
2. **Summary Sheet**: Provides statistics and summary information

**Sample Output Format**:
```
Timestamp: 2025-10-20 08:00
Student_ID: 101
Student_Name: Ahmed Ali
Classroom: Grade 1-A
Status: Present
Similarity_Score: 0.9950
Cutoff_Time: 08:00
```

### Generated Files

- `attendance_report.xlsx` - Main attendance report
- `demo_attendance_report.xlsx` - Demo report with successful match
- `student_database.json` - Student information database
- `student_embeddings/` - Directory containing face embeddings (.npy files)

## Architecture

The system is built with a modular architecture for better maintainability:

- **`main.py`**: Entry point and orchestration
- **`utils.py`**: Model loading and basic utility functions
- **`database.py`**: Student database management and operations
- **`attendance.py`**: Core attendance logic, matching, and reporting
- **`config.py`**: Centralized configuration settings
- **`test_attendance_system.py`**: Testing and demonstration utilities

## How It Works

1. **Face Detection**: YuNet detects faces in the input image
2. **Embedding Extraction**: InsightFace extracts 512-dimensional face embeddings
3. **Face Matching**: Compares detected embeddings with registered student embeddings using cosine similarity
4. **Attendance Classification**: Determines Present/Late status based on detection time vs cutoff time
5. **Report Generation**: Creates Excel reports with attendance records and summary statistics

## File Structure

```
yunet-attendance/
├── main.py                          # Main attendance system entry point
├── utils.py                         # Model loading and utility functions
├── database.py                      # Student database management
├── attendance.py                    # Attendance logic and processing
├── config.py                        # Configuration settings
├── test_attendance_system.py        # Test script with simulated match
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── face_detection_yunet_2023mar.onnx # YuNet face detection model
├── sample_image.jpg                 # Sample test image
├── student_database.json            # Student information database
├── student_embeddings/              # Face embedding files
│   ├── student_101_embedding.npy
│   ├── student_102_embedding.npy
│   └── student_103_embedding.npy
└── venv/                           # Virtual environment
```

## Adding New Students

To add new students to the system:

1. **Add student information** to `student_database.json`:
   ```json
   {
     "id": "104",
     "name": "New Student",
     "classroom": "Grade 1-C",
     "embedding_file": "student_104_embedding.npy"
   }
   ```

2. **Generate face embedding**:
   - Use InsightFace to extract embedding from student's photo
   - Save as `.npy` file in `student_embeddings/` directory
   - Ensure embedding is normalized (512 dimensions)


## Example Output

```
Starting YuNet Attendance System...
Loading YuNet face detection model...
Loading InsightFace buffalo_l model...
Loading student database...
Processing image: sample_image.jpg
Image shape: (4000, 4684, 3)
Detecting faces with YuNet...
Found 1 faces

Processing face 1...
Face 1: Successfully extracted embedding (shape: (512,))
Face 1: Matched to Ahmed Ali (ID: 101)
Similarity score: 0.9950
Attendance status: Present

Exporting 1 attendance records to Excel...
Attendance report exported to attendance_report.xlsx

=== ATTENDANCE SUMMARY ===
2025-10-20 08:00, 101, Ahmed Ali, Grade 1-A, Present

Processing completed successfully!
```