import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Digit Generator", page_icon=":1234:")
st.title(":1234: Handwritten Digit Generator")
st.markdown("Generate handwritten digit images (0-9) using a trained Conditional GAN model.")

# Load trained generator model
@st.cache_resource
def load_generator_model():
    return tf.keras.models.load_model("generator_model.keras")

generator = load_generator_model()

# Constants
LATENT_DIM = 100

def generate_digits(digit, n=5):
    noise = np.random.normal(0, 1, (n, LATENT_DIM))
    labels = np.full((n, 1), digit)
    images = generator.predict([noise, labels], verbose=0)
    return images

# Sidebar selection
selected_digit = st.sidebar.selectbox("Choose a digit to generate:", list(range(10)))

if st.sidebar.button("Generate"):
    with st.spinner("Generating images..."):
        generated_imgs = generate_digits(selected_digit, n=5)

    st.subheader(f"Generated images for digit: {selected_digit}")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(generated_imgs[i].squeeze(), use_column_width=True, clamp=True, channels="GRAY")
else:
    st.info("Click 'Generate' in the sidebar to start.")
