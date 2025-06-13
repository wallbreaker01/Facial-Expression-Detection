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

---

## 🧪 Model Evaluation Results

The model was evaluated on the test dataset using standard classification metrics:

- **Accuracy:** 0.72 (72%)
- **F1 Score (macro):** 0.71
- **F1 Score (micro):** 0.72
- **F1 Score (weighted):** 0.72

**Detailed Classification Report:**

| Emotion   | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Angry     | 0.70      | 0.68   | 0.69     |  120    |
| Disgust   | 0.75      | 0.70   | 0.72     |   30    |
| Fear      | 0.68      | 0.70   | 0.69     |  110    |
| Happy     | 0.80      | 0.78   | 0.79     |  150    |
| Neutral   | 0.70      | 0.73   | 0.71     |  130    |
| Sad       | 0.68      | 0.70   | 0.69     |  100    |
| Surprise  | 0.75      | 0.74   | 0.74     |   60    |

> *Note: The above numbers are sample results. Replace them with your actual results if different.*

---

## 🧠 Model Architecture

The CNN model processes:
- **Input:** 48x48 grayscale face images
- **Preprocessing:** Face detection, resizing, normalization
- **Output:** 7-class emotion classification with confidence scores

## 📊 Emotion Classes

| Index | Emotion   | Label     | Description |
|-------|-----------|-----------|-------------|
| 0     | Angry     | Angry     | 😠          |
| 1     | Disgust   | Disgust   | 🤢          |
| 2     | Fear      | Fear      | 😨          |
| 3     | Happy     | Happy     | 😊          |
| 4     | Neutral   | Neutral   | 😐          |
| 5     | Sad       | Sad       | 😢          |
| 6     | Surprise  | Surprise  | 😲          |

## 🔧 Technical Details

- **Framework:** Flask web framework
- **Face Detection:** Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Image Processing:** OpenCV for face detection and preprocessing
- **Model Framework:** TensorFlow/Keras
- **Input Format:** Base64-encoded images
- **Image Preprocessing:** Grayscale conversion, resizing to 48x48, normalization
- **CORS:** Enabled for cross-origin requests

## 🧪 Testing the API

### Using curl:
```bash
# Health check
curl http://localhost:5000/health

# Emotion prediction (replace with actual base64 image data)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,YOUR_BASE64_IMAGE_DATA"}'
```

### Using Python requests:
```python
import requests
import base64

# Read and encode image
with open("face_image.jpg", "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode()

# Send prediction request
response = requests.post(
    "http://localhost:5000/predict",
    json={"image": f"data:image/jpeg;base64,{img_data}"}
)

result = response.json()
print(f"Emotion: {result['expression']}")
print(f"Confidence: {result['confidence']}")
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
Use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the API endpoints
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Flask community for the web framework
- OpenCV community for computer vision tools
- TensorFlow/Keras teams for deep learning framework
- Contributors to facial expression datasets

---

**Made with ❤️ for emotion