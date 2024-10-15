import cv2
import os
import time

# Define categories
categories = ['trash']

# Create directories for each category if they don't exist
for category in categories:
    if not os.path.exists(category):
        os.makedirs(category)

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit the application.")

# Counter for the images
img_counter = 0
category_index = 0

# Set the interval for capturing (5 seconds)
capture_interval = 5  # seconds
last_capture_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check if 5 seconds have passed since the last capture
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        # Save the image in the current category
        img_name = f"image_{img_counter}.png"
        category = categories[category_index]
        destination = os.path.join(category, img_name)
        cv2.imwrite(destination, frame)
        print(f"{img_name} saved in {category} category.")
        
        img_counter += 1
        last_capture_time = current_time  # Update the last capture time

        # Switch category every 50 images
        if img_counter % 50 == 0:
            category_index += 1
            if category_index >= len(categories):
                print("All categories used up, stopping capture.")
                break

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User requested exit.")
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()