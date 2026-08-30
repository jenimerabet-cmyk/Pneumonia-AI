import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "best_pneumonia_mobilenet.keras"

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter the path to an X-ray image: ")

img = image.load_img(
    image_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array, verbose=0)[0][0]

if prediction >= 0.5:
    label = "PNEUMONIA"
    confidence = prediction
else:
    label = "NORMAL"
    confidence = 1 - prediction

print()
print("Prediction:", label)
print("Confidence:", round(float(confidence) * 100, 2), "%")