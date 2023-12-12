import streamlit as st
import numpy as np
from PIL import Image
!pip install tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

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
