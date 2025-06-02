# 🎭 Facial Expression Detection API

A web-based facial expression recognition system built with Flask and deep learning that can detect and classify seven different emotions from uploaded images.

## ✨ Features

- **REST API**: Flask-based web API for emotion detection
- **Image Upload**: Analyze emotions from uploaded images
- **7 Emotion Classes**: Recognizes angry, disgust, fear, happy, neutral, sad, and surprise
- **Deep Learning Model**: CNN architecture trained on facial expression datasets
- **OpenCV Integration**: Efficient face detection and image processing
- **CORS Support**: Cross-origin requests enabled for frontend integration
- **Health Check**: API status monitoring endpoint

## 🚀 Demo

The API accepts base64-encoded images and returns emotion predictions with confidence scores and bounding box coordinates for detected faces.

## 📁 Project Structure

```
├── app.py                      # Flask API server
├── index.html                  # Frontend web interface
├── emotiondetector.h5          # Primary trained model weights
├── ModelTraining.ipynb         # Training pipeline and model development
├── RealTimeDeployment.ipynb    # Real-time inference implementation
└── images/
    ├── train/                  # Training dataset
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/                   # Test dataset
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

## 🛠️ Installation

### Prerequisites

- Python 3.11.1
- Trained emotion detection model (`.h5` file)

### Dependencies

```bash
pip install flask
pip install flask-cors
pip install opencv-python
pip install tensorflow
pip install numpy
pip install pillow
pip install tensorflow
pip install keras
pip install pandas
pip install jupyter
pip install notebook
pip install tqdm
pip install scikit-learn
```

## 🎯 Usage

### Starting the API Server

1. Ensure you have a trained model file (e.g., `emotiondetector.h5`) in the project directory
2. Run the Flask application:

```bash
python app.py
```

3. The API will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```

Response:
```json
{
  "status": "running",
  "model_loaded": true,
  "model_path": "emotiondetector.h5",
  "expression_labels": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
}
```

#### 2. Emotion Prediction
```http
POST /predict
Content-Type: application/json
```

Request body:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

Response (Success):
```json
{
  "success": true,
  "expression": "Happy",
  "confidence": 0.892,
  "all_predictions": [0.01, 0.02, 0.05, 0.89, 0.02, 0.008, 0.002]
}
```

Response (Error):
```json
{
  "success": false,
  "error": "No face detected in the image"
}
```

#### 3. Home Page
```http
GET /
```

Returns: "Facial Expression Detection API is running!"

## 🧠 Model Architecture

The CNN model processes:
- **Input**: 48x48 grayscale face images
- **Preprocessing**: Face detection, resizing, normalization
- **Output**: 7-class emotion classification with confidence scores

## 📊 Emotion Classes

| Index | Emotion   | Label     | Description |
|-------|-----------|-----------|-------------|
| 0     | Angry     | Angry     | 😠          |
| 1     | Disgust   | Disgust   | 🤢          |
| 2     | Fear      | Fear      | 😨          |
| 3     | Happy     | Happy     | 😊          |
| 4     | Sad       | Sad       | 😢          |
| 5     | Surprise  | Surprise  | 😲          |
| 6     | Neutral   | Neutral   | 😐          |

## 🔧 Technical Details

- **Framework**: Flask web framework
- **Face Detection**: Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Image Processing**: OpenCV for face detection and preprocessing
- **Model Framework**: TensorFlow/Keras
- **Input Format**: Base64-encoded images
- **Image Preprocessing**: Grayscale conversion, resizing to 48x48, normalization
- **CORS**: Enabled for cross-origin requests