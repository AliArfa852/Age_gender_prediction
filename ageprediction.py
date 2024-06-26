# -*- coding: utf-8 -*-
"""AgePrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nfwDIrdd8sGsHzw8ON8rx7xsfW7yimer
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the "utkcropped" folder
path = Path("/content/drive/My Drive/utkcropped")

# Get filenames of JPEG images
filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

# Shuffle filenames
np.random.seed(10)
np.random.shuffle(filenames)

# Initialize lists for labels and image paths
age_labels, gender_labels, image_path = [], [], []

# Define regular expression pattern to extract age and gender
pattern = r"(\d+)_(\d)_\d+_\d+\.jpg\.chip"

# Extract labels from filenames
for filename in filenames:
    match = re.match(pattern, filename)
    if match:
        age_labels.append(int(match.group(1)))
        gender_labels.append(int(match.group(2)))
        image_path.append(str(path / filename))

# Create DataFrame
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_path, age_labels, gender_labels

# Convert gender labels to integer
df['gender'] = df['gender'].astype('int32')

# Convert age labels to float
df['age'] = df['age'].astype('float32')

# Preprocess the images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize images to 224x224 (adjust as needed)
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img

# Load and preprocess images
X = np.array([preprocess_image(image) for image in df['image']])
y = df['age'].values

# Check the size of the dataset
print("Number of samples in the dataset:", len(X))

# Check the first few image paths to verify they are correct
print("First few image paths:", image_path[:5])

# Check if the images are successfully loaded and preprocessed
print("Shape of preprocessed images:", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define the model architecture with added complexity
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1)  # Output layer for age prediction
])

# Compile the model with a lower learning rate and using mean absolute error as the loss function
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model with data augmentation and early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Save the model if needed
model.save("/content/drive/My Drive/age_prediction_model.h5")

"""# ***Testing Cell***"""

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("/content/drive/My Drive/age_prediction_model.h5")

# Function to predict age from an image file
def predict_age(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predicted_age = model.predict(img_array)[0][0]

    return predicted_age

# Example usage:
image_path = "download.jpg"  # Replace with the actual image path
predicted_age = predict_age(image_path)
print("Predicted age:", predicted_age)

import re
import numpy as np
import pandas as pd
from pathlib import Path
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the "utkcropped" folder
path = Path("/content/drive/My Drive/utkcropped")

# Get filenames of JPEG images
filenames = list(map(lambda x: x.name, path.glob('*.jpg')))

# Shuffle filenames
np.random.seed(10)
np.random.shuffle(filenames)

# Initialize lists for labels and image paths
age_labels, gender_labels, image_path = [], [], []

# Define regular expression pattern to extract age and gender
pattern = r"(\d+)_(\d)_\d+_\d+\.jpg\.chip"

# Extract labels from filenames
for filename in filenames:
    match = re.match(pattern, filename)
    if match:
        age_labels.append(int(match.group(1)))
        gender_labels.append(int(match.group(2)))
        image_path.append(str(path / filename))

# Create DataFrame
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_path, age_labels, gender_labels

# Convert gender labels to integer
df['gender'] = df['gender'].astype('int32')

# Convert age labels to float
df['age'] = df['age'].astype('float32')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Define batch size
batch_size = 32

# Flow training images in batches
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image",
    y_col="age",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="raw"
)

# Flow validation images in batches
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image",
    y_col="age",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="raw"
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define the base model (pre-trained VGG16 model)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
base_model.trainable = False

# Define the model architecture with transfer learning
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1)  # Output layer for age prediction
])

# Compile the model with a lower learning rate and using mean absolute error as the loss function
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model with data generators and early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)

# Save the model if needed
model.save("/content/drive/My Drive/age_prediction_model_v2.h5")

"""# **Testing cell 2**"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load the trained model
model = load_model("/content/drive/My Drive/age_prediction_model_v2.h5")

# Function to predict age from an image file
def predict_age(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predicted_age = model.predict(img_array)[0][0]

    return predicted_age

# Example usage:
image_path = "istockphoto-1158015118-612x612.jpg"
predicted_age = predict_age(image_path)
print("Predicted age:", predicted_age)