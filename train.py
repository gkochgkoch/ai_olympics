from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Set the paths for your dataset (Windows paths)
train_dir = r"c:/bin/train_ready"
validation_dir = r"c:/bin/validation"
test_dir = r"c:/bin/test"

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize the images
    rotation_range=20,         # Augmentations (optional)
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load Training and Validation Data
train_generator = train_datagen.flow_from_directory(
    train_dir,                 # Training dataset path
    target_size=(224, 224),    # Resize to 224x224
    batch_size=32,
    class_mode='binary'        # Binary classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,            # Validation dataset path
    target_size=(224, 224),    # Resize to 224x224
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained MobileNetV2 and freeze base layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Build the classification model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce feature maps
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=50, validation_data=validation_generator)

# Preprocess Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,                  # Test dataset path
    target_size=(224, 224),    # Resize to 224x224
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")
model.save(r"c:/bin/trash_bin_model_epoch10.keras")