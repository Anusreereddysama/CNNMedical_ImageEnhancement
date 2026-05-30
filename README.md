# CNN Medical Image Enhancement System

## Overview

This project implements a CNN-based Medical Image Enhancement System using a Convolutional Autoencoder to improve the quality of MRI images. The model learns to reconstruct clean images from noisy inputs, enabling automatic denoising and enhancement of medical scans.

An interactive Streamlit dashboard allows users to upload MRI images, visualize enhancement results, analyze image quality metrics, and download enhanced images in real time.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Scikit-Image
- Pillow

---

## Project Objectives

- Enhance the visual quality of medical images
- Reduce image noise using deep learning techniques
- Preserve important medical structures during enhancement
- Provide real-time enhancement through a web application

---

## Machine Learning Workflow

1. Data Collection
2. Image Preprocessing
3. Noise Generation
4. CNN Autoencoder Training
5. Medical Image Enhancement
6. Performance Evaluation
7. Streamlit Deployment

---

## CNN Architecture

### Encoder
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D

### Bottleneck
- Feature Compression Layer

### Decoder
- Conv2D (64 Filters)
- UpSampling2D
- Conv2D (32 Filters)
- UpSampling2D
- Conv2D Output Layer

---

## Dataset

Dataset Used:

Brain MRI Images Dataset

The dataset consists of MRI brain scan images used for training the autoencoder model to learn image reconstruction and enhancement.

### Preprocessing Steps

- Grayscale Conversion
- Image Resizing (128 × 128)
- Pixel Normalization
- Synthetic Noise Generation
- Creation of Clean and Noisy Image Pairs

---

## Project Structure

```text
CNN_Medical_Image_Enhancement/
│
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── yes/
│   │   └── no/
│   │
│   └── processed/
│       ├── clean/
│       └── noisy/
│
├── models/
│   └── cnn_medical_enhancer.keras
│
└── images/
```

---

## Features

- CNN-Based Medical Image Enhancement
- Medical Image Denoising
- Interactive Streamlit Dashboard
- Original vs Enhanced Image Comparison
- PSNR Evaluation Metric
- SSIM Evaluation Metric
- Pixel Intensity Analysis
- Histogram Visualization
- Download Enhanced Images
- Real-Time Enhancement

---

## Dashboard Features

### Model Information
- CNN Architecture Details
- Dataset Information
- Input Image Specifications

### Image Enhancement
- MRI Image Upload
- Real-Time Enhancement
- Side-by-Side Visualization

### Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Mean Pixel Intensity Analysis

### Visualization
- Histogram Comparison
- Enhancement Analysis Dashboard

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anusreereddysama/CNN_Medical_Image_Enhancement.git
```

Move into the project directory:

```bash
cd CNN_Medical_Image_Enhancement
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

Generate clean and noisy image pairs:

```bash
python preprocess.py
```

---

## Model Training

Train the CNN Autoencoder:

```bash
python train_model.py
```

The trained model will be saved in:

```text
models/cnn_medical_enhancer.keras
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

## Evaluation Metrics

The enhancement quality is assessed using:

### PSNR
Peak Signal-to-Noise Ratio measures image reconstruction quality.

### SSIM
Structural Similarity Index evaluates structural preservation between original and enhanced images.

---

## Future Enhancements

- U-Net Based Enhancement
- Attention Mechanisms
- Transfer Learning
- Advanced Denoising Networks
- Medical Image Segmentation Integration
- Multi-Modal Medical Imaging Support
- Cloud Deployment

---
