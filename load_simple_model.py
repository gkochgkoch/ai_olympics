from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model(r"c:/bin/trash_bin_model_epoch10.keras")

# Load and preprocess a new image for prediction
img = image.load_img(r"c:/bin/new_image.png", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make a prediction
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Full")
else:
    print("Empty")