import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import serial
import time

# Load your trained model
model = load_model('waste_classification_model.h5')

# Define the class labels (merged into two categories)
class_labels = {
    'biodegradable': ['compost', 'paper'],
    'non_biodegradable': ['cardboard', 'glass', 'metal', 'plastic', 'trash']
}

# Start camera capture
cap = cv2.VideoCapture(0)

# Open serial connection to Arduino (adjust 'COM3' to your port)
arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish

# Initialize variables for tracking predictions
start_time = time.time()
predictions = []

def classify_and_control_servo(label):
    """
    Function to send servo control signals based on model prediction
    :param label: Predicted class label from the ML model
    """
    try:
        if label in class_labels['biodegradable']:
            arduino.write(b'biodegradable\n')  # Send command to Arduino
            print("Sent: biodegradable")
        elif label in class_labels['non_biodegradable']:
            arduino.write(b'non_biodegradable\n')  # Send command to Arduino
            print("Sent: non_biodegradable")
    except serial.SerialException as e:
        print(f"Error writing to Arduino: {e}")
    time.sleep(0.1)  # Small delay to allow the servo to move

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match the training input
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224
    img_array = img_to_array(frame_resized)        # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0                  # Normalize pixel values

    # Predict the class of the object in the frame
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    label = list(class_labels['biodegradable'] + class_labels['non_biodegradable'])[class_index]
    confidence = prediction[0][class_index]

    # Store the label and confidence
    if confidence > 0.6:  # Only consider predictions with confidence > 60%
        predictions.append((label, confidence))

    # Every 30 seconds, determine the label with the highest average confidence
    elapsed_time = time.time() - start_time
    if elapsed_time >=10:
        if predictions:
            # Calculate the average confidence for each label
            label_confidences = {}
            for pred_label, conf in predictions:
                if pred_label not in label_confidences:
                    label_confidences[pred_label] = []
                label_confidences[pred_label].append(conf)

            # Determine the label with the highest average confidence
            best_label = max(label_confidences, key=lambda k: np.mean(label_confidences[k]))
            classify_and_control_servo(best_label)

            # Reset tracking variables
            predictions = []
        start_time = time.time()

    # Display the prediction and confidence
    if confidence > 0.6:
        cv2.putText(frame, f"Prediction: {label} ({confidence*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No object detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Waste Classification', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
