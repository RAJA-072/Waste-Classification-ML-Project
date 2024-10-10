import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths to your dataset
train_dir = r'C:\Users\raji\Downloads\Waste-or-Garbage-Classification-Using-Deep-Learning-main (2)\Waste-or-Garbage-Classification-Using-Deep-Learning-main\Waste-or-Garbage-Classification-Using-Deep-Learning-main\DataSets\Train\train'

# ImageDataGenerator for preprocessing and augmenting the data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    shear_range=0.2,  # Apply random shear
    zoom_range=0.2,  # Apply random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2  # Reserve 20% of data for validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = Sequential([
    Input(shape=(224, 224, 3)),  # Explicit Input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes (plastic, paper, metal, cardboard, glass, compost, trash)
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define EarlyStoppQing callback to stop training when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model for up to 50 epochs, with EarlyStopping
history = model.fit(
    train_generator,
    epochs=50,  # Maximum 50 epochs
    validation_data=validation_generator,
    callbacks=[early_stopping]  # Add early stopping callback
)

# Save the trained model
model.save('waste_classification_model.h5')


# Function to choose waste type based on highest priority
def classify_waste(image):
    pred_probabilities = model.predict(image)
    waste_types = ['plastic', 'paper', 'metal', 'cardboard', 'glass', 'compost', 'trash']

    # Get index of highest confidence prediction
    predicted_class = waste_types[pred_probabilities.argmax()]

    print(f"Predicted waste type: {predicted_class}")
    return predicted_class
