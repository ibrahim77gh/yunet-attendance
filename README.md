# YuNet Face Detection + InsightFace Embedding + Excel Export

This script combines YuNet face detection, InsightFace embeddings, and exports results to Excel.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the YuNet model:
   - Download `face_detection_yunet_2023mar.onnx` from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
   - Place it in the same directory as the script

3. Prepare an image:
   - Place a sample image named `sample_image.jpg` in the directory, or
   - Modify the `image_path` variable in `main.py`

## Usage

```bash
python main.py
```

## What it does

1. **YuNet Face Detection**: Detects faces in the input image using the YuNet ONNX model
2. **InsightFace Embeddings**: Extracts face embeddings using the buffalo_l model
3. **Excel Export**: Creates a `.xlsx` file with:
   - Timestamp and image information
   - Face count
   - First 5 embedding dimensions for each detected face
   - Embedding norm values

## Output

The script will create `face_analysis_results.xlsx` with the analysis results.

## Notes

- If no image is provided, the script creates a dummy image for demonstration
- The script handles cases where no faces are detected
- Embedding extraction may fail for very small or unclear faces
