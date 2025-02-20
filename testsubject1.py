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
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]
)

test_datagen = ImageDataGenerator(rescale=1./255)

# --- Load and Preprocess Training Data ---
# Assuming the training images are organized in directories by class
images_train = "./data/Train/"
train_data, train_labels = [], []
classes = 43

# Use flow_from_directory for loading images from folders
train_generator = train_datagen.flow_from_directory(
    images_train,  # Path to training data
    target_size=(128, 128),  # Resize to 128x128 for reduced memory usage
    batch_size=32,  # You can adjust the batch size depending on your system's memory
    class_mode='categorical'
)

# --- Load and Preprocess Test Data ---
# Assuming the test images are organized similarly
images_test_path = "./data/Test/"
test_data = []

for index, row in df.iterrows():
    img_path = os.path.join(images_test_path, row['Path'].strip())
    if os.path.exists(img_path):
        im = Image.open(img_path).resize((128, 128))  # Resize to 128x128
        im = np.array(im)
        test_data.append(apply_red_filter(im))
    else:
        print(f"Warning: Image {img_path} not found!")

X_test = np.array(test_data).astype('float32') / 255.0
y_test = np_utils.to_categorical(test_labels, 43)

# --- Model Definition ---
CNN = Sequential([
    InputLayer(shape=(128, 128, 3)),  # Updated for 128x128 image size
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'), Dropout(0.5),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(43, activation='softmax')
])

CNN.compile(loss=focal_loss(), optimizer='adam', metrics=['accuracy'])

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('stop_sign_model.h5', save_best_only=True, monitor='val_loss')

# --- Model Training ---
CNN.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,  # Number of batches per epoch
    epochs=50,  # You can adjust the number of epochs based on results
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // 32,  # Number of validation batches
    callbacks=[early_stop, model_checkpoint]
)

# Save Model
CNN.save(r"C:\Users\Ricar\Documents\CNN_TF_Stop_Sign_Improved.keras")