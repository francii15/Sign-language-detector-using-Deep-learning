# Sign Language Recognition

A deep learning-based Sign Language Recognition system that classifies **41 static sign-language classes** from images using **MobileNetV2 transfer learning**.

## Project Overview

The model is trained on **18,450 images** across 41 classes. Images are resized to **224 × 224 pixels** and augmented using rotation, zoom, translation, and contrast transformations.

The project uses a two-stage training approach:

1. **Transfer Learning** – MobileNetV2 with ImageNet pretrained weights is used as the feature extractor.
2. **Fine-Tuning** – Later MobileNetV2 layers are selectively unfrozen and trained with a lower learning rate.

The classification head consists of Global Average Pooling, Batch Normalization, a 256-unit Dense layer, Dropout, and a 41-class Softmax output.

## Model Architecture

Input Image (224 × 224 × 3)
        ↓
Data Augmentation
        ↓
MobileNetV2
(ImageNet Weights)
        ↓
Global Average Pooling
        ↓
Batch Normalization
        ↓
Dense Layer (256, ReLU)
        ↓
Dropout (0.40)
        ↓
Dense Layer (41, Softmax)
        ↓
Predicted Sign

## Sign Classes

The model recognizes 41 classes including:

- A–Z
- 1–10
- HELLO
- I_LOVE_YOU
- YES
- NO
- THANK_YOU

## Features

- 41-class sign-language classification
- Image-based sign prediction
- Real-time webcam recognition
- Prediction confidence display
- Top-5 predictions
- Classification report
- Confusion matrix
- OpenCV-based webcam inference
- MediaPipe Hand Landmarker for hand localization during deployment

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab

## Training

The model uses data augmentation techniques including:

- Random rotation
- Random zoom
- Random translation
- Random contrast

Training also uses:

- Adam Optimizer
- Early Stopping
- Model Checkpointing
- Reduce Learning Rate on Plateau

## Inference

The trained model supports both **single-image prediction** and **real-time webcam prediction**.

For webcam inference, OpenCV captures the video frames, converts them to RGB, resizes them to 224 × 224 pixels, and passes them to the trained model for classification.

The prediction is displayed along with its confidence score and FPS.



```text
sign_language_final_41class.keras
sign_language_classes_41.json
