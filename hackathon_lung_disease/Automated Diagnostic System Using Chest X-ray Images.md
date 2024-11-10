
# Automated Diagnostic System Using Chest X-ray Images

## Introduction

This project focuses on developing an automated diagnostic system for chest X-ray images to detect and classify them into two classes (0 and 1) using machine learning techniques. The system leverages a combination of deep learning for image classification using EfficientNetB0 and structured data analysis using basic medical records.

## Data Preparation

### Dataset
- **Image Data**: Chest X-ray images organized into subdirectories (0 and 1).
- **Structured Data**: Basic medical records, including:
  - Image Index
  - Finding Labels
  - Follow-up
  - Patient ID
  - Patient Age
  - Patient Gender
  - View Position
  - Original Image Width and Height
  - Original Image Pixel Spacing (x, y)

### Data Loading and Preprocessing
- **Image Data**:
  - Images were loaded using OpenCV and resized to 224x224 pixels.
  - Normalized pixel values to the range [0, 1].
- **Structured Data**:
  - Normalized numerical features.
  - Encoded categorical variables.
- **Data Splitting**:
  - Split data into training and validation sets for model evaluation.

## Model Architecture

### Base Model
- Pretrained **EfficientNetB0** model (without top layers) from ImageNet.

### Custom Layers
1. Global Average Pooling layer.
2. Dense layer with 512 units (ReLU activation).
3. Batch Normalization layer.
4. Dropout layer (rate: 0.5).
5. Dense output layer with 1 unit (sigmoid activation).

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pretrained EfficientNetB0 base model
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
```

## Training Procedure

### Data Augmentation
- Applied augmentations to improve model robustness:
  - Rotation
  - Width/Height Shift
  - Shear
  - Zoom
  - Horizontal Flip

```python
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
```

### Training Configuration
- **Initial Training**:
  - Epochs: 50
  - Callbacks: EarlyStopping and ReduceLROnPlateau
  - Batch Size: 32
- **Continued Training**:
  - Loaded saved model weights to continue training.
  - Fine-tuned model with a reduced learning rate.

```python
# Load saved weights
weights_path = 'path_to_saved_weights/model_weights.h5'
model.load_weights(weights_path)

# Fit the model
history = model.fit(
    datagen.flow([train_images, X_train], y_train, batch_size=32),
    validation_data=([val_images, X_val], y_val),
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

## Evaluation and Results

### Validation Accuracy
- Achieved initial validation accuracy of approximately **76%**.
- Further improved accuracy through fine-tuning.

### Final Model Evaluation
```python
val_loss, val_acc = model.evaluate(val_images, y_val)
print(f"Validation Accuracy after additional training: {val_acc:.2f}")
```

## Conclusion

The automated diagnostic system for chest X-ray classification was successfully developed using a combination of EfficientNetB0 and custom layers. Data augmentation and fine-tuning significantly improved model performance, achieving a final validation accuracy of approximately **76%**. Further improvements are possible with additional training, hyperparameter tuning, and ensemble methods.

