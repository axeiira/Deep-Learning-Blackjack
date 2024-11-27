import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore
import os

# Parameters
img_height = 128
img_width = 128 
batch_size = 32  
epochs = 10       

# Load and preprocess the dataset
data_dir = 'datasets'

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Reserve 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Generate class mapping
class_indices = train_generator.class_indices  # Get class indices
class_mapping_file_path = 'class_mapping.txt'  # Path to save the mapping file

with open(class_mapping_file_path, 'w') as f:
    for class_name, class_index in class_indices.items():
        f.write(f"{class_index}: {class_name}\n")  # Write class index and name

print(f"Class mapping saved to {class_mapping_file_path}")

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the trained model
model.save('model/cnn_model.h5')
print("Model saved as cnn_model.h5")

# Optional: plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
