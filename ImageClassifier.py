import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('digit_classifier.h5')

def load_image(image_file):
    img = Image.open(image_file)
    return img


def convert_rgba_to_rgb(image):
    """Convert RGBA image to RGB by pasting it onto a white background."""
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    return new_image.convert('RGB')


def resize_and_invert(image):
    """Resize the image to 28x28 and invert its colors."""
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    return ImageOps.invert(image)


def normalize_and_reshape(image):
    """Normalize the image pixel values and reshape for model input."""
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return np.expand_dims(image_array, axis=0)  # Add batch dimension


def preprocess_image(image):
    """Preprocess an image for model input."""
    if image.mode == 'RGBA':
        image = convert_rgba_to_rgb(image)

    image = ImageOps.grayscale(image)
    image = resize_and_invert(image)
    return normalize_and_reshape(image)

st.title('Digit Classifier App')

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    processed_image = preprocess_image(image)

    if st.button('Predict'):
        prediction = model.predict(processed_image)
        st.write(f'Predicted Digit: {np.argmax(prediction)}')
