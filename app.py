from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your trained model
try:
    # Add your actual model filename to the list
    model_files = [
        'emotiondetector.h5',  # Your actual model file
        'emotion_model.h5',
        'facial_expression_model.h5', 
        'model.h5',
        'emotiondetection.h5',
        'fer_model.h5'
    ]
    
    model_path = None
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break
    
    if model_path:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    else:
        print("No model file found. Please check your model path.")
        print("Looking for files:", model_files)
        print("Current directory files:", [f for f in os.listdir('.') if f.endswith('.h5')])
        model = None
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Expression labels - make sure these match your model's training order
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return "Facial Expression Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict_expression():
    try:
        print("Received prediction request")
        
        if model is None:
            print("Model is not loaded!")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check model path.'
            })
            
        # Get image data from request
        data = request.json
        print("Processing image data...")
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            })
            
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image decoded successfully: {image.size}")
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        print(f"Detected {len(faces)} faces")
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            (x, y, w, h) = largest_face
            print(f"Face coordinates: x={x}, y={y}, w={w}, h={h}")
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to model input size (48x48 for most emotion models)
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Normalize pixel values
            face_roi = face_roi.astype('float32') / 255.0
            
            # Reshape for model input: (1, 48, 48, 1)
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            print(f"Face ROI shape: {face_roi.shape}")
            print(f"Face ROI min/max: {face_roi.min():.3f}/{face_roi.max():.3f}")
            
            # Predict expression
            predictions = model.predict(face_roi, verbose=0)
            print(f"Raw predictions: {predictions[0]}")
            
            expression_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][expression_idx])
            expression = expression_labels[expression_idx]
            
            print(f"Prediction: {expression} (index: {expression_idx}) with confidence: {confidence:.3f}")
            
            return jsonify({
                'success': True,
                'expression': expression,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist()
            })
        else:
            print("No face detected in image")
            return jsonify({
                'success': False,
                'error': 'No face detected in the image'
            })
            
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_path': model_path if model is not None else None,
        'expression_labels': expression_labels
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model loaded: {model is not None}")
    if model is not None:
        print(f"Model path: {model_path}")
    app.run(debug=True, port=5000, host='0.0.0.0')