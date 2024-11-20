import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# Load your trained model
model = load_model('CNN_TF.keras')  # or 'my_model.h5'

# Stream URL of the ESP32-CAM
stream_url = 'http://192.168.4.2/stream'

# Open the stream using OpenCV
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create directories to save logs and correct predictions
logs_dir = 'logs'
output_dir = 'correct_predictions'

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    # Read a frame from the stream
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to read frame.")
        break
    
    # Resize the frame to match the model's input size
    img_resized = cv2.resize(frame, (48, 48))

    # Normalize the image
    img_resized = img_resized / 255.0

    # Add an extra dimension to match the expected input shape
    img_resized = np.expand_dims(img_resized, axis=0)

    # Make the prediction
    prediction = model.predict(img_resized)
    prediction_class = np.argmax(prediction, axis=1)

    # Timestamp for logging
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Log the prediction with a timestamp
    log_file_name = os.path.join(logs_dir, f'prediction_log_{timestamp}.txt')
    with open(log_file_name, 'w') as log_file:
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Predicted class: {prediction_class[0]}\n")
        log_file.write(f"Prediction details: {prediction}\n")

    # If the prediction is correct, save the image to a folder
    if prediction_class[0] == 1:  # Modify this condition based on the correct class you're checking for
        # Create a filename for saving the image (with timestamp)
        image_filename = os.path.join(output_dir, f'correct_prediction_{timestamp}.jpg')
        
        # Save the image
        cv2.imwrite(image_filename, frame)
        print(f"Correct prediction! Image saved as {image_filename}")
    
    # Overlay the predicted class on the image
    cv2.putText(frame, f'Prediction: {prediction_class[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with prediction
    cv2.imshow("Live Stream", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the stream and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
