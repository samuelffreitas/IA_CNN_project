# Traffic Sign Recognition with Convolutional Neural Networks

A machine learning project to classify traffic signs using a Convolutional Neural Network (CNN), based on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

---

## 📖 Project Overview

This project uses Convolutional Neural Networks (CNNs) to recognize and classify traffic signs. Advanced image preprocessing techniques and CNN architectures are applied to identify patterns and ensure accurate classification. While the focus is on recognizing **STOP signs** (Class 14 in the dataset), this project also explores computer vision techniques and neural network concepts. 

It includes implementations using TensorFlow/Keras and other Python libraries to preprocess, train, and evaluate the model. The main goal is to build a robust classification system with high accuracy for traffic sign detection.

---

## ⚙️ Installation

Follow these steps to set up and run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/samuelffreitas/IA_CNN_project.git
   ```

2. Navigate to the project directory:
   ```bash
   cd IA_CNN_project
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

- `TensorFlow` / `Keras` (for deep learning and CNN training)
- `NumPy` (for array and numerical operations)
- `Matplotlib` (for visualization)
- `Scikit-learn` (for metrics and preprocessing)
- `OpenCV` (for image processing)

Make sure Python 3.7+ is installed.

4. Download the **GTSRB dataset** [here](https://benchmark.ini.rub.de/gtsrb_dataset.html) and place the data in the project folder under a directory named `data/`.
5. Extract the dataset and paste the contents in a new folder named "data".
6. Delete the two .csv files inside the "Test" folder.

---

## 📊 Data Preprocessing

The following preprocessing steps are applied to prepare the dataset:

1. **Image Resizing**: All images are resized to a uniform size of **48x48 pixels** to maintain consistency and reduce computation.
2. **Normalization**: Pixel values are normalized to the range [0, 1] by dividing by 255. This helps in faster convergence during training.
3. **Categorical Encoding**: The target labels are one-hot encoded for multi-class classification.

---

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) to classify traffic signs. 

### CNN Architecture
- **Input Layer**: Accepts 48x48 pixel images with 3 color channels (RGB).
- **Convolutional Layers**: Two convolutional layers with ReLU activation to extract features.
  - Convolution + Max Pooling operations to reduce dimensionality.
- **Fully Connected Layers**: Two dense layers for classification.
- **Output Layer**: Softmax activation to predict probabilities for 43 classes.

While the CNN is trained for all 43 categories in the GTSRB dataset, the project primarily focuses on detecting **STOP signs (Class 14)**.

---

## 🏋️ Training

1. **Loss Function**: Categorical Crossentropy is used to measure the model's error.
2. **Optimizer**: Adam optimizer with a learning rate of 0.001 for efficient training.
3. **Epochs and Batch Size**: The model is trained for 5 epochs with a batch size of 128 (adjustable).
4. **Train-Test Split**: The dataset is already split between train and test.

---

## 📋 Results

- **Training Accuracy**: ~91.44%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: Achieved high precision in classifying STOP signs.

The model successfully identifies STOP signs from the dataset, making it suitable for real-world traffic sign detection applications.

---

## 🧪 Potential Enhancements

- **Real-Time Detection**: Integrate a camera feed to detect STOP signs in real-time.
- **More Classes**: Extend the focus beyond STOP signs to classify all 43 categories..

---

## 📂 Directory Structure

```
KNN_Bitcoin-Heist/
├── .gitignore               # Specifies files and directories to ignore in version control
├── DatasetReference.bib     # Bibliography reference for the dataset
├── README.md                # Project documentation
├── main.ipynb               # Jupyter Notebook with main script
├── requirements.txt         # List of dependencies for the project
```

---

## References

- [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- TensorFlow/Keras documentation

---
