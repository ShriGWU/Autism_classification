# Autism_classification
# 🧠 Autism Detection from Brain MRI Scans using Deep Learning

## 🔍 Overview

This project implements a deep learning pipeline to classify **Autism Spectrum Disorder (ASD)** from structural brain MRI images. The workflow involves preprocessing `.nii` scans, applying data augmentation, training a 3D CNN/ResNet model, and deploying the system using a full-stack approach.

- 🚀 **Frontend**: User interface via [Lovable](https://lovable.so)
- 🔌 **Backend**: Flask API for uploading & predicting MRI scans
- 🖥️ **Server**: Google Cloud VM with NVIDIA L4 GPU
- 🧠 **Dataset**: ABIDE (Autism Brain Imaging Data Exchange)

---

## 🧠 Dataset

- **Source**: [ABIDE MRI Dataset](http://fcon_1000.projects.nitrc.org/indi/abide/)
- **Classes**: 
  - `Autistic` (Treatment) → 547 scans
  - `Control` (Non-ASD) → 617 scans
- **Total Before Augmentation**: 1164 `.nii` MRI scans
- **After 3x Augmentation on 80% Training Data**:
  - ~4,368 total training images
  - ~233 per class in test set (20%)

---

## 🧪 Preprocessing Steps

1. **Conversion**: `.nii` → `numpy` arrays (`.npy`)
2. **Normalization**: Scaled pixel intensities to range `[0, 1]`
3. **Resizing**: Volumes resized to `(80, 128, 128)`
4. **Batch Processing**: Training and labels saved as `.npy` batch files

---

## 🔁 Augmentation (Training Set Only)

Augmentation was applied **only to training samples** using `tf.image` methods:

- `random_flip_left_right()`
- `random_brightness()` (delta=0.1)
- `random_contrast()` (0.9 - 1.1)
- 3x augmented per original training image

---

## 🧠 Model Architecture

### ➤ 3D Residual CNN

- `Conv3D` + `BatchNorm` + `ReLU`
- Custom residual blocks
- `MaxPooling3D`, `GlobalAveragePooling3D`
- Final output layer: `Dense(1, activation='sigmoid')`

```python
input → Conv3D → Residual Blocks → GAP → Dense → Sigmoid
