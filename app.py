import streamlit as st
import numpy as np
import imageio
import torch
from filters import butterworth_lowpass_filter, anisotropic_diffusion, median_filter, bilateral_filter_color, gaussian_filter
from denoise_model import load_image, load_dncnn_model, denoise_image

st.title("Image Filtering & Denoising with Streamlit")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = imageio.imread(uploaded_file)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

    # Select filter type
    filter_choice = st.selectbox("Choose a Filter", [
        "Butterworth Low-Pass", 
        "Anisotropic Diffusion", 
        "Median Filter", 
        "Bilateral Filter",
        "Gaussian Filter",
        "DnCNN (Deep Learning) Denoising"
    ])

    if filter_choice == "Butterworth Low-Pass":
        cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=200, value=60)
        if st.button("Apply Filter"):
            filtered_image = butterworth_lowpass_filter(image, cutoff)

    elif filter_choice == "Anisotropic Diffusion":
        iterations = st.slider("Iterations", min_value=5, max_value=50, value=20)
        kappa = st.slider("Kappa", min_value=10, max_value=100, value=30)
        gamma = st.slider("Gamma", min_value=0.05, max_value=0.5, value=0.2)
        option = st.radio("Diffusion Function", [1, 2])
        if st.button("Apply Filter"):
            filtered_image = anisotropic_diffusion(image, iterations, kappa, gamma, option)

    elif filter_choice == "Median Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=5, step=2)
        if st.button("Apply Filter"):
            filtered_image = median_filter(image, filter_size)

    elif filter_choice == "Bilateral Filter":
        d = st.slider("Filter Window Size", min_value=3, max_value=15, value=9, step=2)
        sigma_s = st.slider("Spatial Sigma", min_value=1, max_value=50, value=10)
        sigma_r = st.slider("Range Sigma", min_value=1, max_value=100, value=25)
        if st.button("Apply Filter"):
            filtered_image = bilateral_filter_color(image, d, sigma_s, sigma_r)

    elif filter_choice == "Gaussian Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=3, step=2)
        sigma = st.slider("Sigma Value", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        if st.button("Apply Filter"):
            filtered_image = gaussian_filter(image, filter_size, sigma)

    elif filter_choice == "DnCNN (Deep Learning) Denoising":
        model_path = "model.pth"  # Path to your trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if st.button("Apply DnCNN Model"):
            try:
                model = load_dncnn_model(model_path, device)
                noisy_tensor, noisy_array = load_image(uploaded_file)
                filtered_image = denoise_image(model, noisy_tensor, device)
            except Exception as e:
                st.error(f"Error: {e}")
                filtered_image = image  # Return original if error

    # Display results
    if "filtered_image" in locals():
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(filtered_image, caption="Filtered Image", use_column_width=True)
