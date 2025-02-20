import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import keras
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Custom Focal Loss Function ---
from keras.saving import register_keras_serializable

@register_keras_serializable()
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

# --- Load and Process Data ---
df = pd.read_csv('./data/Test.csv')
df.columns = df.columns.str.strip()
df = df.drop(columns=['Width', 'Height', 'Roi.X1', 'Roi.X2', 'Roi.Y2', 'Roi.Y1'], axis=1)
test_labels = df['ClassId'].to_numpy().flatten()

# Preprocessing Functions
def apply_red_filter(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel = image_rgb[:, :, 0]
    red_mask = red_channel > 100
    filtered_image = np.zeros_like(image_rgb)
    filtered_image[:, :, 0] = red_channel * red_mask
    return filtered_image

def apply_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, threshold1=100, threshold2=200)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, 
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)

# Load and Preprocess Training Data
images_train = "./data/Train/"
train_data, train_labels = [], []
classes = 43

for i in range(classes):
    img_path = os.path.join(images_train, str(i))
    for img in os.listdir(img_path):
        im = Image.open(img_path + '/' + img).resize((48, 48))
        im = np.array(im)
        train_data.append(apply_red_filter(im))
        train_labels.append(i)

X_train = np.array(train_data).astype('float32') / 255.0
y_train = np_utils.to_categorical(np.array(train_labels), 43)

# Load and Preprocess Test Data
images_test_path = "./data/Test/"
test_data = []

for index, row in df.iterrows():
    img_path = os.path.join(images_test_path, row['Path'].strip())
    if os.path.exists(img_path):
        im = Image.open(img_path).resize((48, 48))
        im = np.array(im)
        test_data.append(apply_red_filter(im))
    else:
        print(f"Warning: Image {img_path} not found!")

X_test = np.array(test_data).astype('float32') / 255.0
y_test = np_utils.to_categorical(test_labels, 43)

# --- Model Definition ---
CNN = Sequential([
    InputLayer(shape=(48, 48, 3)),
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(43, activation='softmax')
])

CNN.compile(loss=focal_loss(), optimizer='adam', metrics=['accuracy'])

# Train Model
datagen.fit(X_train)
CNN.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=5)

# Save Model
CNN.save(r"C:\Users\Ricar\Documents\CNN_TF_Stop_Sign_Improved.keras")