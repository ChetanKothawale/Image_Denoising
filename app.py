import streamlit as st
import numpy as np
import imageio
import torch
from filters import anisotropic_diffusion, median_filter, bilateral_filter_color, gaussian_filter
from DnCNN_filter import load_image, load_dncnn_model, denoise_image
from gan_model import load_gan_model, preprocess_image, denoise_image as gan_denoise_image

# Set up Streamlit UI
st.title("Denoising Noisy Images with Deep Learning & Traditional Filters")

uploaded_file = st.file_uploader("Upload a Noisy Image", type=["png", "jpg", "jpeg"])

# Load Deep Learning Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DnCNN model
dncnn_model_path = "model.pth"
try:
    dncnn_model = load_dncnn_model(dncnn_model_path, device)
    st.success("DnCNN Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading DnCNN model: {e}")

# Load GAN model
gan_model_path = "generator_model.pth"
try:
    gan_model = load_gan_model(gan_model_path)
    st.success("GAN Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading GAN model: {e}")

if uploaded_file:
    # Read and display the noisy image
    noisy_image = imageio.imread(uploaded_file)

    if noisy_image.ndim == 2:
        noisy_image = np.stack([noisy_image] * 3, axis=-1)  # Convert grayscale to RGB

    st.image(noisy_image, caption="Noisy Image", use_column_width=True)

    # Select denoising method
    denoise_choice = st.selectbox(
        "Choose a Denoising Method",
        [
            "Anisotropic Diffusion",
            "Median Filter",
            "Bilateral Filter",
            "Gaussian Filter",
            "DnCNN (Deep Learning)",
            "GAN-Based Denoising"
        ],
    )

    # Apply selected denoising method
    if denoise_choice == "Anisotropic Diffusion":
        iterations = st.slider("Iterations", min_value=5, max_value=50, value=20)
        kappa = st.slider("Kappa", min_value=10, max_value=100, value=30)
        gamma = st.slider("Gamma", min_value=0.05, max_value=0.5, value=0.2)
        option = st.radio("Diffusion Function", [1, 2])
        if st.button("Denoise Image"):
            denoised_image = anisotropic_diffusion(noisy_image, iterations, kappa, gamma, option)

    elif denoise_choice == "Median Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=5, step=2)
        if st.button("Denoise Image"):
            denoised_image = median_filter(noisy_image, filter_size)

    elif denoise_choice == "Bilateral Filter":
        d = st.slider("Filter Window Size", min_value=3, max_value=15, value=9, step=2)
        sigma_s = st.slider("Spatial Sigma", min_value=1, max_value=50, value=10)
        sigma_r = st.slider("Range Sigma", min_value=1, max_value=100, value=25)
        if st.button("Denoise Image"):
            denoised_image = bilateral_filter_color(noisy_image, d, sigma_s, sigma_r)

    elif denoise_choice == "Gaussian Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=3, step=2)
        sigma = st.slider("Sigma Value", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        if st.button("Denoise Image"):
            denoised_image = gaussian_filter(noisy_image, filter_size, sigma)

    elif denoise_choice == "DnCNN (Deep Learning)":
        if st.button("Denoise Image"):
            try:
                noisy_tensor, _ = load_image(uploaded_file)
                denoised_image = denoise_image(dncnn_model, noisy_tensor, device)
            except Exception as e:
                st.error(f"Error: {e}")
                denoised_image = noisy_image  # Return original if error

    elif denoise_choice == "GAN-Based Denoising":
        if st.button("Denoise Image"):
            try:
                noisy_tensor = preprocess_image(uploaded_file)
                denoised_image = gan_denoise_image(gan_model, noisy_tensor)
            except Exception as e:
                st.error(f"Error: {e}")
                denoised_image = noisy_image  # Return original if error

    # Display results
    if "denoised_image" in locals():
        col1, col2 = st.columns(2)
        with col1:
            st.image(noisy_image, caption="Noisy Image", use_column_width=True)
        with col2:
            st.image(denoised_image, caption="Denoised Image", use_column_width=True)
