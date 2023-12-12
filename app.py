import subprocess
subprocess.call(["pip", "install", "tensorflow==2.14.0"])
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import os

# Load model architecture from JSON file
model_json_path = os.path.join(os.path.dirname(__file__), 'model.json')

with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read('model.json')

print("Loaded Model JSON:")
print(loaded_model_json)

# Load the model from JSON
model = tf.keras.models.model_from_json(loaded_model_json)

# Close the JSON file
model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into the new model
model.load_weights("model.h5")

# Function to preprocess the image for diagnosis
def preprocess_image(img_path):
    IMM_SIZE = 224  # Update with your desired size

    # Initialize image variable
    image = None

    try:
        # Attempt to read the image from the file
        image = Image.open(img_path)
    except Exception as e:
        # Print an error message if the image cannot be read
        st.error(f"Error reading image from {img_path}: {e}")
        return None

    # Resize the image to the desired size
    image = image.resize((IMM_SIZE, IMM_SIZE))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Function to make a diagnosis using the provided function
def make_diagnosis(img_path):
    diag = diagnosis(img_path)
    return diag

# Streamlit app
st.title("Medical Image Diagnosis App")

uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    if img_array is not None:
        # Make diagnosis
        diagnosis_result = make_diagnosis(img_array)

        # Display the result
        st.write("Diagnosis:")
        st.write(diagnosis_result)
