 
# **Waste Classification System with Real-Time Camera and Servo Control**

![image](https://github.com/user-attachments/assets/a44ae485-0960-466c-aea1-8f1439cd3812)



This project integrates a machine learning-based waste classification system with real-time camera processing and Arduino-controlled servo motors. It classifies waste into **biodegradable** and **non-biodegradable** categories and controls hardware actions based on the classification
---

## **Features**
- **Machine Learning Model**:
  - Built using TensorFlow/Keras with a CNN model for classifying waste.
  - Classifies waste into categories like `plastic`, `paper`, `metal`, `cardboard`, `glass`, `compost`, and `trash`.
  - Maps waste categories into `biodegradable` and `non_biodegradable`

- **Real-Time Camera Integration**:
  - Captures video feed from a computer or external camera using OpenCV.
  - Preprocesses frames in real-time for model predictions

- **Arduino Servo Control**:
  - Communicates with an Arduino via serial to control servos for waste sorting.
  - Sends commands like `biodegradable` or `non_biodegradable` to Arduino based on predictions.

- **Dynamic Decision Logic**:
  - Implements a confidence threshold to ensure accurate predictions.
  - Processes predictions over a time interval for better reliability.

---

## **Requirements**
### **Software**
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- NumPy
- pySerial
- A pre-trained TensorFlow model (`waste_classification_model.h5`).

### **Hardware**
- A camera (webcam or external camera).
- Arduino with a servo motor attached.
- A system capable of running Python scripts.

  ![WhatsApp Image 2024-11-16 at 15 58 04_acac5948](https://github.com/user-attachments/assets/7aa58d11-3014-41a0-9372-49a558028556)

---

## **Setup**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/waste-classification.git
   cd waste-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow opencv-python numpy pyserial
   ```

3. **Prepare the Model**:
   - Place the pre-trained TensorFlow model (`waste_classification_model.h5`) in the project directory.
   - If you don't have a model, train one using the dataset and code provided.

4. **Connect Arduino**:
   - Upload a compatible Arduino sketch for servo control to the Arduino board.
   - Connect the board to your computer via USB.

5. **Run the Script**:
   ```bash
   python waste_classification.py
   ```

---

## **Usage**
1. **Camera Feed**:
   - The script captures real-time video frames from the connected camera.
   - Displays the frame with prediction results overlayed.

2. **Waste Classification**:
   - The ML model predicts the type of waste with confidence levels.
   - The predictions are shown on the video feed.

3. **Servo Motor Control**:
   - Based on classification, the system sends signals to the Arduino to trigger servo actions.

4. **Stop the Program**:
   - Press `q` to quit the real-time video feed and end the program safely.

---

## **Directory Structure**
```
waste-classification/
├── waste_classification.py    # Main script for the project
├── waste_classification_model.h5 # Pre-trained TensorFlow model
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── arduino_sketch/            # Arduino code for servo control
```

---

## **Troubleshooting**
- **Camera Not Detected**:
  - Ensure the camera is properly connected and recognized by your system.
- **Model File Not Found**:
  - Verify the `waste_classification_model.h5` is in the project directory.
- **Arduino Connection Fails**:
  - Check the COM port and baud rate in the script.
- **Prediction Accuracy Low**:
  - Fine-tune the model with a larger dataset or adjust the confidence threshold.

---

## **Future Enhancements**
- Integrate a mobile app for remote monitoring.
- Use advanced models like VGG16 or ResNet for improved accuracy.
- Add sound alerts for classification categories.

---

## **Acknowledgments**
- TensorFlow for providing robust machine learning frameworks.
- OpenCV for real-time image processing capabilities.
- Arduino for enabling seamless hardware integration

---
