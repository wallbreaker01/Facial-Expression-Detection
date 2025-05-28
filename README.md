# 🎭 Facial Expression Detection

A real-time facial expression recognition system built with deep learning that can detect and classify seven different emotions from live webcam feed.

## ✨ Features

- **Real-time Detection**: Live emotion recognition from webcam feed
- **7 Emotion Classes**: Recognizes angry, disgust, fear, happy, neutral, sad, and surprise
- **Deep Learning Model**: CNN architecture trained on facial expression datasets
- **OpenCV Integration**: Efficient face detection and image processing
- **User-friendly Interface**: Simple webcam interface with visual feedback

## 🚀 Demo

The system detects faces in real-time and overlays the predicted emotion on the video feed with bounding boxes around detected faces.

## 📁 Project Structure

```
├── emotiondetector.h5          # Trained model weights
├── emotiondetector.json        # Model architecture
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

- Python 3.7+
- Webcam/Camera access

### Dependencies

```bash
pip install opencv-python
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install pillow
pip install tqdm
```

## 🎯 Usage

### Training the Model

1. Open and run [`ModelTraining.ipynb`](ModelTraining.ipynb) to train the emotion detection model
2. The notebook will:
   - Load and preprocess the training data
   - Build a CNN architecture with multiple convolutional layers
   - Train the model for emotion classification
   - Save the trained model as `emotiondetector.h5` and `emotiondetector.json`

### Real-time Emotion Detection

1. Open [`RealTimeDeployment.ipynb`](RealTimeDeployment.ipynb)
2. Run all cells to start the real-time detection system
3. The system will:
   - Load the pre-trained model
   - Access your webcam
   - Detect faces in real-time
   - Classify emotions and display results
4. Press `q` or `ESC` to quit the application

## 🧠 Model Architecture

The CNN model consists of:
- **Input Layer**: 48x48 grayscale images
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: 7 neurons with softmax activation for emotion classification

## 📊 Emotion Classes

| Index | Emotion   | Description |
|-------|-----------|-------------|
| 0     | Angry     | 😠          |
| 1     | Disgust   | 🤢          |
| 2     | Fear      | 😨          |
| 3     | Happy     | 😊          |
| 4     | Neutral   | 😐          |
| 5     | Sad       | 😢          |
| 6     | Surprise  | 😲          |

## 🔧 Technical Details

- **Face Detection**: Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Image Preprocessing**: Grayscale conversion, resizing to 48x48, normalization
- **Model Framework**: TensorFlow/Keras
- **Real-time Processing**: OpenCV for video capture and display

## 🎮 Controls

- **Q key**: Quit the application
- **ESC key**: Exit the program
- **Webcam**: Automatic face detection and emotion recognition

## 📈 Performance

The model is trained to recognize facial expressions with high accuracy across different lighting conditions and face orientations. The real-time system processes video frames efficiently for smooth user experience.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow/Keras teams for deep learning framework
- Contributors to facial expression datasets

---

**Made with ❤️ for emotion recognition research and applications**