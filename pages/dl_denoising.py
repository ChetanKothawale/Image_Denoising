import streamlit as st
import numpy as np
import torch
from dncnn import load_image, load_model, denoise_image
from PIL import Image

st.title("Deep Learning Image Denoising (DnCNN)")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pth"  # Ensure this file is in the root directory
model = load_model(model_path, device)

uploaded_file = st.file_uploader("Upload a Noisy Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_tensor, image_array = load_image(uploaded_file)
    
    if st.button("Denoise Image"):
        denoised_img = denoise_image(model, image_tensor, device)
        
        # Convert back to PIL Image for display
        denoised_pil = Image.fromarray((denoised_img * 255).astype(np.uint8))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_array, caption="Noisy Image", use_column_width=True, channels="L")
        with col2:
            st.image(denoised_pil, caption="Denoised Image", use_column_width=True, channels="L")
