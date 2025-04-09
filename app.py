import streamlit as st
import numpy as np
import imageio
import matplotlib.pyplot as plt

def butterworth_lowpass_filter(image, cutoff_freq, n=8):
    """Apply a Butterworth low-pass filter in the frequency domain to a color image."""
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    for channel in range(image.shape[2]):
        fft = np.fft.fft2(image[:, :, channel])
        fft_shifted = np.fft.fftshift(fft)

        rows, cols = image.shape[:2]
        mask = np.zeros((rows, cols))
        center = (rows // 2, cols // 2)

        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                mask[i, j] = 1 / (1 + (D / cutoff_freq) ** (2 * n))

        filtered = fft_shifted * mask
        fft_inverse = np.fft.ifftshift(filtered)
        filtered_image[:, :, channel] = np.fft.ifft2(fft_inverse).real

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return filtered_image

st.title("Low-Pass Filtering with Streamlit")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = imageio.imread(uploaded_file)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

    cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=200, value=60)
    
    if st.button("Apply Low-Pass Filter"):
        filtered_image = butterworth_lowpass_filter(image, cutoff)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(filtered_image, caption="Filtered Image", use_column_width=True)
