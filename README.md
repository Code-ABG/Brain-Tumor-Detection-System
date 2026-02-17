🧠 Brain Tumor MRI Classification using Transfer Learning
📌 Overview

This project implements a deep learning-based multi-class Brain Tumor classification system using MRI images. The model classifies brain scans into four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The system uses MobileNetV2 with transfer learning and fine-tuning, achieving 93.36% validation accuracy. Grad-CAM is implemented for explainable AI to visualize tumor regions influencing predictions.

🚀 Key Features

Transfer Learning using MobileNetV2

Fine-tuning with low learning rate optimization

Multi-class classification (4 classes)

Data augmentation for improved generalization

Confusion Matrix and Classification Report evaluation

Grad-CAM visualization for explainability

🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

OpenCV

📂 Dataset

Brain Tumor MRI Dataset containing:

Training images

Testing images

4 tumor classes

Images were resized to 150x150 and normalized before training.

🧠 Model Architecture

Pretrained MobileNetV2 (ImageNet weights)

Global Average Pooling

Dense layer (ReLU activation)

Dropout for regularization

Softmax output layer (4 classes)

📊 Model Performance
Metric	Value
Validation Accuracy	93.36%
Validation Loss	0.1828

Model evaluation includes:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

🔍 Explainability (Grad-CAM)

Grad-CAM is used to generate heatmaps highlighting image regions responsible for predictions. This improves model interpretability in medical imaging tasks.

📈 Training Strategy

Baseline CNN implementation

Transfer learning with frozen layers

Fine-tuning deeper layers

Reduced learning rate (1e-5) during fine-tuning

Data augmentation to prevent overfitting

▶️ How to Run

Clone the repository

Install required dependencies

Mount Google Drive (if using Colab)

Run the notebook step-by-step

Use the prediction function to classify new MRI images

📌 Future Improvements

Deploy as web application (Streamlit/Flask)

Improve class balancing

Experiment with EfficientNet

Cross-validation for robustness

🧑‍💻 Author

Abhinav Gupta
Aspiring AI/ML Engineer
