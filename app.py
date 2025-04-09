import streamlit as st
import numpy as np
import imageio
from filters import butterworth_lowpass_filter, anisotropic_diffusion

st.title("Image Filtering with Streamlit")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = imageio.imread(uploaded_file)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

    # Select filter type
    filter_choice = st.selectbox("Choose a Filter", ["Butterworth Low-Pass", "Anisotropic Diffusion"])

    if filter_choice == "Butterworth Low-Pass":
        cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=200, value=60)
        if st.button("Apply Filter"):
            filtered_image = butterworth_lowpass_filter(image, cutoff)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(filtered_image, caption="Filtered Image", use_column_width=True)

    elif filter_choice == "Anisotropic Diffusion":
        iterations = st.slider("Iterations", min_value=5, max_value=50, value=20)
        kappa = st.slider("Kappa (Edge Threshold)", min_value=10, max_value=100, value=30)
        gamma = st.slider("Gamma (Step Size)", min_value=0.05, max_value=0.5, value=0.2)
        option = st.radio("Diffusion Function", [1, 2])
        
        if st.button("Apply Filter"):
            filtered_image = anisotropic_diffusion(image, iterations, kappa, gamma, option)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(filtered_image, caption="Filtered Image", use_column_width=True)
