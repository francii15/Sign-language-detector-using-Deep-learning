# 🤟 Signa – Real-Time Sign Language Recognition

**Signa** is a real-time Sign Language Recognition application built as an exploration of **Deep Learning and Computer Vision**.

The goal of this project was to explore how well a deep learning model could learn visual patterns of hand signs directly from image data and then move that model beyond a notebook into a real-time camera application.

Rather than building the recognition primarily around traditional machine learning or a MediaPipe hand-landmark pipeline, this project focuses on an **image-based deep learning approach using MobileNetV2 and Transfer Learning**.

---

## 💡 Overview

Signa captures frames from a live camera, preprocesses them, passes them through a trained deep learning model, and displays the predicted sign along with its confidence.

The project evolved from model experimentation in Jupyter Notebook into a complete real-time application with a browser-based interface built using Streamlit.

---

## ✨ Features

- 🎥 Real-time webcam sign recognition
- 🤟 Recognition of **41 sign classes**
- 📊 Live prediction confidence
- 🧠 Deep Learning based image classification
- 🔄 Prediction smoothing for a more stable live experience
- 🌐 Browser-based camera access
- 💻 Clean Streamlit user interface
- ⚡ Real-time inference

---

## 🧠 Deep Learning Approach

This project was intentionally developed as an exploration of **deep learning for visual sign recognition**.

Instead of primarily extracting predefined hand landmarks and training a traditional machine-learning classifier, the model learns visual features directly from sign images.

The core model uses:

- **MobileNetV2**
- **ImageNet pretrained weights**
- **Transfer Learning**
- **Fine-tuning**
- **Global Average Pooling**
- **Dense layers**
- **Dropout**
- **41-class Softmax output**

MobileNetV2 was selected because it provides a good balance between image-recognition capability and computational efficiency, which is useful when moving toward real-time inference.

---

## 🤟 Supported Signs

Signa currently supports **41 classes**.

### Numbers
`1, 2, 3, 4, 5, 6, 7, 8, 9, 10`

### Alphabets
`A – Z`

### Common Expressions
- `HELLO`
- `YES`
- `NO`
- `THANK_YOU`
- `I_LOVE_YOU`

---

## 📊 Dataset

The dataset contains:

- **18,450 images**
- **41 classes**
- **450 images per class**

Image augmentation was used during training to help the model learn from variations in the training images.

---

## 🏗️ Model Architecture

The recognition pipeline is based on MobileNetV2 with transfer learning.

```text
Input Image
     ↓
MobileNetV2
(ImageNet pretrained)
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
41-Class Softmax Output
     ↓
Predicted Sign
```

The model was trained using a two-stage approach involving transfer learning followed by fine-tuning.

---

## 🔄 How Signa Works

```text
Live Camera
     ↓
Capture Frame
     ↓
Image Preprocessing
     ↓
Deep Learning Model
     ↓
Class Probabilities
     ↓
Prediction + Confidence
     ↓
Prediction Smoothing
     ↓
Signa Interface
```

The browser camera provides frames to the application. Each frame selected for inference is prepared for the trained model, which predicts one of the supported sign classes.

The latest prediction and confidence are then displayed through the Signa interface.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Architecture | MobileNetV2 |
| Learning Approach | Transfer Learning |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Web Application | Streamlit |
| Real-Time Camera | Streamlit WebRTC |
| Development | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```text
Sign-language-detector-using-Deep-learning/
│
├── app.py
├── requirements.txt
├── README.md
├── Sign_language_project.ipynb
├── Sign-Language-Recognition (1) (1).pptx
│
└── models/
    ├── sign_language_final_41class.keras
    └── sign_language_classes_41.json
```

### Important Files

**`app.py`**  
Runs the Signa Streamlit application and real-time recognition pipeline.

**`requirements.txt`**  
Contains the Python dependencies required to run the application.

**`Sign_language_project.ipynb`**  
Contains the model development, training, evaluation, and experimentation workflow.

**`sign_language_final_41class.keras`**  
The trained 41-class deep learning model.

**`sign_language_classes_41.json`**  
Stores the class labels used to map model outputs to sign names.

---

## 🚀 Run Signa Locally

### 1. Clone the repository

```bash
git clone https://github.com/francii15/Sign-language-detector-using-Deep-learning.git
```

### 2. Enter the project directory

```bash
cd Sign-language-detector-using-Deep-learning
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Signa

```bash
streamlit run app.py
```

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open it in your browser and allow camera permission.

---

## 🎯 From Training to Real-Time Recognition

One of the most useful parts of this project was seeing the difference between achieving good results on a prepared image dataset and running the same model against frames coming from a real webcam.

Real-world inference introduces additional challenges such as:

- Lighting changes
- Different backgrounds
- Hand positioning
- Camera quality
- Distance from the camera
- Similar-looking signs
- Differences between training images and live-camera images

This made Signa more than just a model-training exercise—it became an exploration of taking a deep learning model from **experimentation to a usable real-time application**.

---

## ⚠️ Current Limitations

Signa currently performs **single-frame image classification**, so it is best suited to static signs.

Recognition can also be affected by:

- Poor lighting
- Busy backgrounds
- Incorrect hand positioning
- Partially visible hands
- Visually similar signs
- Differences between training and real-world camera data

Some signs with similar visual characteristics can therefore be more difficult for the model to distinguish consistently.

---

## 🔮 Future Improvements

Possible improvements include:

- Hand detection and localization before classification
- More diverse real-world training data
- Better handling of uncertain/unknown gestures
- Improved prediction stability
- Model optimization for faster inference
- Temporal recognition for motion-based signs
- Testing lightweight deployment formats for improved performance

A future version could also combine deep learning image recognition with hand localization while keeping the trained classifier as the core recognition model.

---

## 🌐 Live Demo

**Coming soon — Signa will be deployed as a public web application.**

---

## 👤 Author

**Francis Infant**

Built as a hands-on exploration of **Deep Learning, Computer Vision, Transfer Learning, Real-Time Inference, and Web Deployment**.# Signa – Real-Time Sign Language Recognition

Short project introduction

## Overview
Why you built Signa and the deep-learning exploration

## Features
- Real-time webcam recognition
- 41 sign classes
- Prediction confidence
- Streamlit interface
- Browser-based camera

## Deep Learning Approach
- MobileNetV2
- Transfer Learning
- Image-based recognition
- Why you explored DL rather than primarily using a traditional ML/MediaPipe pipeline

## Supported Signs
Numbers + alphabets + HELLO / YES / NO / THANK_YOU / I_LOVE_YOU

## Dataset
18,450 images
41 classes
450 images per class

## Model Architecture
MobileNetV2 → Global Average Pooling → Dense → Dropout → Output

## How Signa Works

Camera
   ↓
Frame
   ↓
Preprocessing
   ↓
Deep Learning Model
   ↓
Prediction
   ↓
Confidence
   ↓
Signa UI

## Application Preview
I screenshot
C screenshot
L screenshot

## Technologies
Python
TensorFlow / Keras
MobileNetV2
OpenCV
NumPy
Streamlit
WebRTC

## Project Structure
app.py
requirements.txt
models/
notebook
etc.

## Run Locally
conda environment / pip installation
streamlit run app.py

## Results

## Challenges & Limitations
Lighting
Background
Similar gestures
Camera/domain shift
Static-image recognition

## Future Improvements
Hand localization
Unknown/uncertain class
Temporal gestures
Performance optimization

## Live Demo
To be added after deployment

## Author
Francis Infant
