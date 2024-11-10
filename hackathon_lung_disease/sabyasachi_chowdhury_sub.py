
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from IPython import display

# Define the correct paths to your structured data files in Google Drive using raw strings
train_file_path = r'/content/drive/My Drive/hackathon_diagonostic/train_set.csv'
test_file_path = r'/content/drive/My Drive/hackathon_diagonostic/test_set.csv'

# Load the structured data into DataFrames
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Display the first few rows of the DataFrame to inspect its contents
display.display(train_data.head())
display.display(train_data.info())
display.display(train_data.describe())

from sklearn.preprocessing import LabelEncoder

# Encode Patient Gender
train_data['Patient Gender'] = LabelEncoder().fit_transform(train_data['Patient Gender'])
test_data['Patient Gender'] = LabelEncoder().fit_transform(test_data['Patient Gender'])

# Encode View Position
train_data['View Position'] = LabelEncoder().fit_transform(train_data['View Position'])
test_data['View Position'] = LabelEncoder().fit_transform(test_data['View Position'])

display.display(train_data.head())

#Normalize Numerical Features:

from sklearn.preprocessing import StandardScaler

# Define the features to be normalized
features_to_scale = ['Patient Age', 'Follow-up', 'OriginalImage_Width', 'OriginalImage_Height', 'OriginalImagePixelSpacing_x', 'OriginalImagePixelSpacing_y']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features in the training and test data
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

# Extract the target variable and feature matrix
y_train = train_data['Finding Labels']
X_train = train_data.drop(columns=['Finding Labels', 'Image Index'])

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# #Extract and Unzip Training Images:
# # import zipfile
# import os

# # Define the path to the training images zip file
# train_images_zip_path = r'/content/drive/My Drive/hackathon_diagonostic/Copy of train_images.zip'

# # Extract the zip file
# with zipfile.ZipFile(train_images_zip_path, 'r') as zip_ref:
#     zip_ref.extractall('/content/drive/My Drive/hackathon_diagonostic/train_images')

# import zipfile
# import os

# # Define the path to the test images zip file
# test_images_zip_path = r'/content/drive/My Drive/hackathon_diagonostic/Copy of test_images.zip'

# # Extract the zip file
# with zipfile.ZipFile(test_images_zip_path, 'r') as zip_ref:
#     zip_ref.extractall('/content/drive/My Drive/hackathon_diagonostic/test_images2')

import matplotlib.pyplot as plt
import cv2
import os
from random import sample

# Path to the extracted training image folder
train_image_folder_path = '/content/drive/My Drive/hackathon_diagonostic/train_images/train_images'
# Function to display random images from the dataset
def display_random_images(df, image_folder_path, n=5):
    # Sample n random rows from the DataFrame
    random_rows = df.sample(n)

    # Set up the matplotlib figure and axes based on the number of images
    fig, axes = plt.subplots(1, n, figsize=(15, 10))

    # If only one image, we need to convert axes to a list
    if n == 1:
        axes = [axes]

    # Go through each random row, read the image, and display it
    for ax, (_, row) in zip(axes, random_rows.iterrows()):
        # Determine the class subdirectory (0 or 1)
        class_subdir = str(row['Finding Labels'])
        # Full path to the image
        img_path = os.path.join(image_folder_path, class_subdir, row['Image Index'])

        # Check if the image path exists
        if os.path.exists(img_path):
            # Read and convert the image from BGR to RGB
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            # Display the image
            ax.imshow(image)
            ax.axis('off')  # Turn off axis
            ax.set_title(f"Class: {class_subdir}")  # Set the title to the class
        else:
            # Display a placeholder if the image does not exist
            ax.text(0.5, 0.5, 'Image Not Found', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')  # Turn off axis

    plt.show()

# Example usage of the function
display_random_images(train_data, train_image_folder_path, n=5)

import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Path to the extracted training image folder
train_image_folder_path = '/content/drive/My Drive/hackathon_diagonostic/train_images/train_images/'

# Function to load and preprocess images
def load_and_preprocess_image(row, image_folder_path, target_size=(224, 224)):
    class_subdir = str(row['Finding Labels'])  # Determine the class subdirectory (0 or 1)
    img_path = os.path.join(image_folder_path, class_subdir, row['Image Index'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = np.stack((img,)*3, axis=-1)  # Convert grayscale to RGB by duplicating channels
        img = img / 255.0  # Normalize pixel values
        return img, row['Finding Labels']
    else:
        return None, None

# Function to parallelize image loading and preprocessing
def process_images_in_parallel(dataframe, image_folder_path, target_size=(224, 224), max_workers=8):
    images = []
    labels = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_and_preprocess_image, row, image_folder_path, target_size)
            for _, row in dataframe.iterrows()
        ]

        for future in tqdm(futures, total=len(futures)):
            img, label = future.result()
            if img is not None:
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load and preprocess all training images in parallel
train_images, y_train = process_images_in_parallel(train_data, train_image_folder_path)

# Reshape images to add channel dimension
train_images = train_images.reshape(train_images.shape[0], 224, 224, 3)

# Split images into training and validation sets
train_images, val_images, y_train, y_val = train_test_split(train_images, y_train, test_size=0.2, random_state=42)

print(f"Number of training images: {train_images.shape[0]}")
print(f"Number of validation images: {val_images.shape[0]}")

!pip install keras-tuner -q

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the CNN model for image processing
image_input = Input(shape=(224, 224, 3))
cnn_base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)
cnn_out = Flatten()(cnn_base.output)
cnn_out = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.05))(cnn_out)
cnn_out = BatchNormalization()(cnn_out)
cnn_out = Dropout(0.7)(cnn_out)  # Increased dropout
cnn_out = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(cnn_out)
cnn_out = BatchNormalization()(cnn_out)
cnn_out = Dropout(0.7)(cnn_out)  # Increased dropout

# Define the neural network for structured data
structured_input = Input(shape=(X_train.shape[1],))
structured_out = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(structured_input)
structured_out = BatchNormalization()(structured_out)
structured_out = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(structured_out)
structured_out = BatchNormalization()(structured_out)

# Combine both models
combined = Concatenate()([cnn_out, structured_out])
combined_out = Dense(1, activation='sigmoid')(combined)

# Create the final model
model = Model(inputs=[image_input, structured_input], outputs=combined_out)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Summary of the model
model.summary()

# Data augmentation for images to improve model robustness
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model with data augmentation
history = model.fit(
    datagen.flow([train_images, X_train], y_train, batch_size=32),
    validation_data=([val_images, X_val], y_val),
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate([val_images, X_val], y_val)
print(f"Validation Accuracy: {val_acc:.2f}")

# Continue training the model with more epochs
additional_epochs = 20  # Define the number of additional epochs

history = model.fit(
    datagen.flow([train_images, X_train], y_train, batch_size=32),
    validation_data=([val_images, X_val], y_val),
    epochs=additional_epochs,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate([val_images, X_val], y_val)
print(f"Validation Accuracy after additional training: {val_acc:.2f}")

# Save the model weights
import os # imports the os module

# Create the directory if it doesn't exist
os.makedirs('path_to_save_weights', exist_ok=True)

model.save_weights('path_to_save_weights/model_weights.h5')

# # Load the model weights
# model.load_weights('path_to_save_weights/model_weights.h5')

## Iteration 1

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the path to the saved weights file
weights_path = 'path_to_save_weights/model_weights.h5'

# Load the pretrained EfficientNetB0 model without the top layers
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

# Load the saved weights into the base model
base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)  # Adding BatchNormalization layer
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Define callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Fit the model
history = model.fit(
    datagen.flow(train_images, y_train, batch_size=32),
    validation_data=(val_images, y_val),
    epochs=20,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_images, y_val)
print(f"Validation Accuracy after additional training: {val_acc:.2f}")

# Save the model weights
model.save_weights('path_to_save_weights/fine_tuned_model_weights.h5')

# Fit the model
history = model.fit(
    datagen.flow(train_images, y_train, batch_size=32),
    validation_data=(val_images, y_val),
    epochs=20,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_images, y_val)
print(f"Validation Accuracy after additional training: {val_acc:.2f}")







import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Path to the extracted test image folder
test_image_folder_path = '/content/drive/My Drive/hackathon_diagonostic/test_images2/test_images'

# Function to load and preprocess images
def load_and_preprocess_test_image(image_name, image_folder_path, target_size=(224, 224)):
    img_path = os.path.join(image_folder_path, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.stack((img,)*3, axis=-1)  # Convert grayscale to RGB by duplicating channels
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img

# Function to parallelize image loading and preprocessing
def process_test_images_in_parallel(image_names, image_folder_path, target_size=(224, 224), max_workers=8):
    images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_and_preprocess_test_image, image_name, image_folder_path, target_size)
            for image_name in image_names
        ]

        for future in tqdm(futures, total=len(futures)):
            img = future.result()
            images.append(img)

    return np.array(images)

# Load and preprocess all test images in parallel
test_image_names = test_data['Image Index']
test_images = process_test_images_in_parallel(test_image_names, test_image_folder_path)

# Reshape images to add channel dimension
test_images = test_images.reshape(test_images.shape[0], 224, 224, 3)

X_test = test_data.drop(columns=['Image Index'])

# Make predictions on test images using only the image data
predictions = model.predict(test_images)

# Convert predictions to binary classes (0 or 1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'Image Index': test_data['Image Index'],
    'Finding Labels': predicted_classes
})

# Save the submission file
submission_file_path = '/content/drive/My Drive/hackathon_diagonostic/submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")



