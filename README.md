# Kidney-Disease-Classification-VGG16
---
Deep-learning project for classifying kidney disease (e.g., cysts, stones, tumors, normal) using a VGG16 transfer-learning model. The project includes data preprocessing, model preparation, model training, and evaluation scripts.
---

## Overview

Kidney disease detection from medical imaging (CT/Ultrasound) helps in early diagnosis and treatment.  
This project leverages **VGG16**, a powerful convolutional neural network pretrained on ImageNet, and fine-tunes it for kidney image classification tasks such as:

- **Normal**
- **Tumor**
- **Stone**
- **Cyst**

Transfer learning enables strong performance even with limited medical datasets.

---

## Features

- Pretrained **VGG16** backbone (ImageNet)
- Transfer learning + fine-tuning
- Image augmentation pipeline
- Model evaluation in MLFlow using Dagshub
- Clean modular code structure

---

## Publicly available datasets can be used, such as:

- Kaggle Kidney Disease Dataset  
- Ultrasound/CT kidney images  
- Custom clinical image dataset

The kaggle downloader script, specifically takes in the kaggle source to download the dataset, which can be customized for another source.

---

## Model Architecture

The model uses **VGG16 (pretrained on ImageNet)** as the feature extractor.

Layers:

- Preprocessing (VGG16 preprocess_input)
- VGG16 Base model
- Flatten 
- Dense Layer (ReLU)  
- Dropout (regularization)  
- Dense Softmax Output Layer (for multi-class classification)

---

## Results

The hyperparameters can be customized in config.yaml. 
Based on different values for learning rate, epochs and dropout rate, the accuracy varied between 90-98% fordifferent combinations.
General trend for good accuracy - learning rate range (1e-3 - 1e-5), epoch > 10 (small dataset ~ 2000 images), dropout rate < 0.4

