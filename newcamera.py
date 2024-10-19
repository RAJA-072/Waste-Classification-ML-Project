import cv2
import numpy as np
import tensorflow as tf
import serial
import time
from collections import Counter

# Load the trained model
model = tf.keras.models.load_model('image_classification_model11.h5')

# Define the class labels
class_labels = ['paper', 'cardboard', 'metal', 'trash', 'plastic']

# Parameters for image preprocessing
img_width, img_height = 128, 128  # Change to 128x128 to match the trained model

# Initialize serial communication with Arduino (Adjust 'COM4' to your Arduino port)
arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize the frame to the required input size of the model
    frame_resized = cv2.resize(frame, (img_width, img_height))
    # Normalize the pixel values
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match the model's input shape
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables for tracking predictions
start_time = time.time()
predictions = []

# Loop for live detection
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)

        # Perform prediction
        prediction = model.predict(processed_frame)
        class_index = np.argmax(prediction[0])  # Get the index of the highest probability
        confidence = prediction[0][class_index]  # Get the confidence level
        predicted_class = class_labels[class_index]  # Get the label of the predicted class

        # Store the label if confidence is above threshold
        if confidence >= 0.5:  # Adjust confidence threshold as needed
            predictions.append(predicted_class)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Every 20 seconds, determine the most frequent prediction
        if elapsed_time >= 15:
            if predictions:
                # Count the frequency of each prediction
                most_common_prediction, count = Counter(predictions).most_common(1)[0]

                # Send the most common prediction to Arduino
                if most_common_prediction == "metal":
                    arduino.write(b'biodegradable\n')  # Signal for Metal
                    print("Sent 'metal' to Arduino.")
                else:
                    arduino.write(b'non_biodegradable\n')  # Signal for Non-Metal
                    print("Sent 'non_metal' to Arduino.")

                # Reset predictions list
                predictions = []
            start_time = time.time()  # Reset the timer

        # Display the prediction and confidence
        cv2.putText(frame, f"Prediction: {predicted_class} ({confidence*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with the prediction
        cv2.imshow('Waste Classification', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()  # Close the serial connection to Arduino
    print("Webcam released and windows closed.")
