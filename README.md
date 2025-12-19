# Brain Tumor Detection using MRI Images

## 📋 Project Overview

This project implements a machine learning model to detect and classify brain tumors from MRI images. The model can identify whether a tumor is present and, if so, classify it into one of three types: Glioma, Meningioma, or Pituitary tumor. This is a deep learning-based medical image classification project designed for educational and research purposes.

**⚠️ Important Note**: This project is for educational and research purposes only and should NOT be used for actual clinical diagnosis.

---

## 🎯 Project Objectives

- Develop a CNN-based model for brain tumor classification
- Compare binary classification (tumor vs. no tumor) and multi-class classification performance
- Learn end-to-end implementation of an image classification pipeline
- Build a professional ML project structure

---

## 📊 Dataset Information

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/akrashnoor/brain-tumor/data)

### Dataset Structure:
- **Total Images**: 9,279 MRI scans (9,271 .jpg + 2 .png + 6 .jpeg)
- **Image Type**: Grayscale, preprocessed MRI scans
- **Resolution**: Standardized for consistency

### Classes:

**Binary Classification (2 Classes)**:
- **Yes** (Tumor Present): 4,159 images
- **No** (No Tumor): 2,024 images

**Multi-class Classification (4 Classes)**:
- **Glioma Tumor**: Brain tumors arising from glial cells
- **Meningioma Tumor**: Tumors in the meninges (brain membranes)
- **Pituitary Tumor**: Tumors in the pituitary gland
- **Normal**: No tumor present

---

## 🗂️ Project Structure

```
brain-tumor-detection/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed images (if applicable)
│   └── README.md              # Dataset documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── model.py               # Model architecture
│   ├── train.py               # Training script
│   └── utils.py               # Helper functions
│
├── models/
│   └── saved_models/          # Trained model checkpoints
│
├── results/
│   ├── plots/                 # Visualization outputs
│   └── metrics/               # Performance metrics
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
└── LICENSE                    # License information
```

---

## 🔧 Requirements

### System Requirements:
- **OS**: WSL Ubuntu / Linux / Windows / macOS
- **Hardware**: CPU (GPU optional but not required)
- **Python**: 3.8 or higher

### Python Libraries:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow (or pytorch)
opencv-python
Pillow
jupyter
```

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On WSL/Linux/Mac
# venv\Scripts\activate   # On Windows

# OR using conda
conda create -n brain-tumor python=3.9
conda activate brain-tumor
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/akrashnoor/brain-tumor/data)
2. Extract and place in the `data/raw/` directory

---

## 💻 Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Explore the dataset, visualize sample images, and understand class distribution.

### 2. Data Preprocessing
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```
Perform image preprocessing, augmentation, and train-test split.

### 3. Model Training
```bash
jupyter notebook notebooks/03_model_training.ipynb
```
Train the CNN model on the preprocessed dataset.

### 4. Model Evaluation
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```
Evaluate model performance using accuracy, confusion matrix, and classification report.

---

## 🧠 Model Architecture

*[To be updated as the project progresses]*

The model uses a Convolutional Neural Network (CNN) architecture optimized for medical image classification. Key features include:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification

---

## 📈 Results

*[To be updated after training]*

### Performance Metrics:
- **Accuracy**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1-Score**: TBD

### Sample Predictions:
*[Visualizations will be added here]*

---

## 🔮 Future Improvements

- [ ] Implement transfer learning with pre-trained models (VGG16, ResNet, EfficientNet)
- [ ] Add data augmentation techniques to improve model generalization
- [ ] Develop a web application for model deployment
- [ ] Experiment with different architectures and hyperparameters
- [ ] Add explainability features (Grad-CAM visualization)
- [ ] Optimize model for CPU inference

---

## 📚 Learning Resources

As this is my first large-scale image project, here are resources I found helpful:
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Medical Image Classification Guide](https://www.pyimagesearch.com/)

---

## 🤝 Contributing

This is a learning project, but suggestions and feedback are welcome! Feel free to:
- Open an issue for bugs or suggestions
- Submit pull requests for improvements
- Share your own implementations or variations

---

## 📄 License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgments

- **Dataset Source**: [Akrashnoor's Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/akrashnoor/brain-tumor/data)
- Thanks to the Kaggle community for providing this valuable dataset
- This project is developed for educational purposes as part of my journey in Machine Learning and AI

---

## 📧 Contact

For questions or collaboration opportunities, feel free to reach out:
- **GitHub**: [@yourusername](https://github.com/init-nj)
- **LinkedIn**: [Your Name](https://linkedin.com/in/initnj)

---

## ⚠️ Disclaimer

This model is developed for educational and research purposes only. It should NOT be used for:
- Clinical diagnosis
- Medical decision-making
- Patient treatment planning

Always consult qualified healthcare professionals for medical advice and diagnosis.
