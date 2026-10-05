# Signa – Real-Time Sign Language Recognition

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
