import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import keras
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Data Preprocessing ---
# Apply red filter to isolate red regions (for Stop Sign)
def apply_red_filter(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel = image_rgb[:, :, 0]  # Extract red channel
    red_mask = red_channel > 100  # Threshold to isolate red regions
    filtered_image = np.zeros_like(image_rgb)
    filtered_image[:, :, 0] = red_channel * red_mask  # Keep only red channel
    filtered_image[:, :, 1] = 0  # Set green and blue channels to 0
    filtered_image[:, :, 2] = 0
    return filtered_image

# Apply Canny edge detection
def apply_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    return edges

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,    # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    shear_range=0.2,        # Random shearing
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True,   # Random horizontal flip
    fill_mode='nearest'     # Fill missing pixels with nearest value
)

# --- Loading and Preprocessing Data ---
images_train = "./data/Train/"
train_data = []
train_labels = []
classes = 43

# Loop through images and apply preprocessing
for i in range(classes):
    img_path = os.path.join(images_train, str(i))
    for img in os.listdir(img_path):
        im = Image.open(img_path + '/' + img)
        im = im.resize((48, 48))  # Resize to 48x48
        im = np.array(im)
        
        # Apply Red Filter and Edge Detection
        im_filtered = apply_red_filter(im)
        im_edges = apply_edge_detection(im)
        
        train_data.append(im_filtered)
        train_labels.append(i)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# --- Test Data Preprocessing ---
images_test_path = "./data/Test/"
test_data = []
test_labels = []

# Loop through the test images
for img in os.listdir(images_test_path):
    im = Image.open(images_test_path + '/' + img)
    im = im.resize((48, 48))  # Resize to 48x48
    im = np.array(im)
    
    # Apply Red Filter and Edge Detection to test data
    im_filtered = apply_red_filter(im)
    im_edges = apply_edge_detection(im)
    
    test_data.append(im_filtered)

test_data = np.array(test_data)

# --- Normalization ---
X_train = train_data.astype('float32') / 255.0
X_test = test_data.astype('float32') / 255.0

# --- One-Hot Encoding for Labels ---
y_train = np_utils.to_categorical(train_labels, 43)
y_test = np_utils.to_categorical(test_labels, 43)

# --- Model Creation ---
CNN = Sequential()

CNN.add(InputLayer(shape=(48, 48, 3)))  # Input layer

# First Conv Layer
CNN.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # Convolution
CNN.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling

# Second Conv Layer
CNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # Convolution
CNN.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling

CNN.add(Flatten())  # Flatten the output from the Conv layers

# Fully Connected Layers (Dense layers)
CNN.add(Dense(units=256, activation='relu'))
CNN.add(Dense(units=128, activation='relu'))
CNN.add(Dense(units=43, activation='softmax'))  # Output layer (43 classes)

CNN.summary()

# --- Class Weight Calculation ---
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: class_weights[i] for i in range(classes)}

# --- Focal Loss Definition ---
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        alpha_t = y_true * alpha + (1. - y_true) * (1. - alpha)
        p_t = y_true * y_pred + (1. - y_true) * (1. - y_pred)
        fl = -tf.reduce_sum(alpha_t * tf.pow((1. - p_t), gamma) * tf.math.log(p_t))
        return fl
    return focal_loss_fixed

# --- Model Compilation ---
CNN.compile(loss=focal_loss(gamma=2., alpha=0.25), optimizer='adam', metrics=['accuracy'])

# --- Model Training with Augmentation ---
datagen.fit(X_train)
CNN.fit(datagen.flow(X_train, y_train, batch_size=128),
        epochs=5, 
        class_weight=class_weight_dict)

# --- Evaluate the Model ---
result = CNN.evaluate(X_test, y_test)
print(f"Test loss: {result[0]}, Test accuracy: {result[1]}")

# --- Save the Model ---
CNN.save('CNN_TF_Stop_Sign_Improved.keras')
