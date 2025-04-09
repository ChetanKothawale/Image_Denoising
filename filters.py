import numpy as np

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

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def anisotropic_diffusion(img, iterations=20, kappa=30, gamma=0.20, option=1):
    """Apply anisotropic diffusion for noise removal."""
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    padded = np.pad(img, ((1,1), (1,1), (0,0)), mode='reflect')

    for _ in range(iterations):
        delta_n = padded[:-2, 1:-1] - padded[1:-1, 1:-1]
        delta_s = padded[2:, 1:-1] - padded[1:-1, 1:-1]
        delta_e = padded[1:-1, 2:] - padded[1:-1, 1:-1]
        delta_w = padded[1:-1, :-2] - padded[1:-1, 1:-1]

        if option == 1:
            c_n = np.exp(-(delta_n/kappa)**2)
            c_s = np.exp(-(delta_s/kappa)**2)
            c_e = np.exp(-(delta_e/kappa)**2)
            c_w = np.exp(-(delta_w/kappa)**2)
        else:
            c_n = 1 / (1 + (delta_n/kappa)**2)
            c_s = 1 / (1 + (delta_s/kappa)**2)
            c_e = 1 / (1 + (delta_e/kappa)**2)
            c_w = 1 / (1 + (delta_w/kappa)**2)

        out = img + gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

        img = out.copy()
        padded = np.pad(img, ((1,1), (1,1), (0,0)), mode='reflect')

    return np.clip(out, 0, 255).astype(np.uint8)
