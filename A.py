import re
import numpy as np
import pandas as pd
from pathlib import Path
# from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# Define the path to the "utkcropped" folder
path = Path("./utkcropped/")

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

# Define data generators
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
    y_col=["age", "gender"],
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="multi_output"
)

# Flow validation images in batches
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image",
    y_col=["age", "gender"],
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="multi_output"
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define the model architecture
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
    Dense(1, name='age_output'),  # Output layer for age prediction
    Dense(1, activation='sigmoid', name='gender_output')  # Output layer for gender prediction
])

# Compile the model with appropriate loss functions and metrics
model.compile(optimizer='adam', loss={'age_output': 'mean_absolute_error', 'gender_output': 'binary_crossentropy'}, 
              loss_weights={'age_output': 1., 'gender_output': 0.5}, metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)

# Evaluate the model
loss, age_mae, gender_loss, gender_acc = model.evaluate(validation_generator)
print("Total Loss:", loss)
print("Age MAE:", age_mae)
print("Gender Loss:", gender_loss)
print("Gender Accuracy:", gender_acc)

# Save the model if needed
model.save("age_and_gender_prediction_model.h5")
