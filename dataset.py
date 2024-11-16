import cv2
import os

# Define categories
categories = ['plastic', 'paper', 'compost', 'trash', 'metal', 'cardboard', 'glass']

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for the user to press a key
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # Press 'ESC' to exit
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # Press 'SPACE' to capture the image
        img_name = "image_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

        # Ask for the category
        print("Enter the category for the image:")
        for i, category in enumerate(categories):
            print(f"{i}: {category}")

        category_idx = int(input().strip())
        if 0 <= category_idx < len(categories):
            category = categories[category_idx]
            # Move the image to the category directory
            os.rename(img_name, os.path.join(category, img_name))
            print(f"Image saved to {category} category.")
        else:
            print("Invalid category index, image discarded.")
            os.remove(img_name)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
