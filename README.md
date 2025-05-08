
---

## 📊 Dataset

- **Source**: [ABIDE (Autism Brain Imaging Data Exchange)](http://fcon_1000.projects.nitrc.org/indi/abide/)
- **Classes**:
  - `Autistic (1)` - 547 images
  - `Control (0)` - 617 images
- **Augmentation**: Training data augmented 3x using TensorFlow's `tf.image`
- **Final Distribution**:
  - After augmentation: 1312 (autistic) + 1480 (control)
  - Train-test split: 80:20 (Stratified)

---

## ⚙️ Preprocessing

- Read `.nii` MRI files using `nibabel`
- Normalize voxel intensities to [0, 1]
- Resize to `(80, 128, 128)` using `scipy.ndimage.zoom`
- Convert to `.npy` format for efficient loading
- Batched and saved into memory-mapped `.npy` files

---

## 🧪 Augmentation

Only training data is augmented using `tf.image` methods:

- `tf.image.random_flip_left_right`
- `tf.image.random_brightness`
- `tf.image.random_contrast`
- `tf.image.random_crop`
- `tf.image.random_saturation`

---

## 🧠 Model

A custom 3D ResNet-inspired CNN:

- Residual blocks with `Conv3D`, `BatchNorm`, `ReLU`
- MaxPooling and GlobalAveragePooling
- Dropout Regularization
- Final Dense Layer with Sigmoid for binary classification

---

## 🖥️ Infrastructure

- **Frontend**: Built using [Lovable](https://lovable.so)
- **Backend**: `Flask` API hosted on GCP VM
- **Model**: Served using TensorFlow 2.15 on NVIDIA L4 GPU
- **Storage**: MRI files stored in Google Cloud Storage Buckets

---

## 🚀 Deployment

1. Upload `.nii` file via frontend
2. Flask API accepts request and saves to `uploads/`
3. Image is preprocessed → reshaped → batched
4. TensorFlow model makes a prediction
5. Result (Autism Detected / Not Detected + Confidence %) is returned

---

## ✅ Evaluation

- Final Test Accuracy: **57%**
- Binary classification: `Autism` vs `Control`
- Evaluation metric: `Accuracy = (TP + TN) / (TP + FP + FN + TN)`

---

## 🔗 Reference Article

[Autism Detection from Brain MRI and Screening Data — A Cloud-Based Machine Learning Approach](https://medium.com/@aparnashankar004/autism-detection-from-brain-mri-and-screening-data-a-cloud-based-machine-learning-approach-5549d582ad45)

---

## 🧠 Future Work

- Integrate behavioral/survey data
- Use pretrained 3D medical models (e.g., MedicalNet)
- Explore explainability (GradCAM for MRI)
- Extend to multi-class neurological prediction

---

## 👤 Author

- **Name**: Shrishail Ravi Terni
- **Role**: Individual project under Master's Program
- **Platform**: Google Cloud (Vertex AI, Compute Engine)

