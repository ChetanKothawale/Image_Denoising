import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the DnCNN model
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=kernel_size//2))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual

# Function to load and preprocess the image
def load_image(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img_tensor, img_array  # Return tensor for processing and array for display

# Function to load the trained model
def load_dncnn_model(model_path, device):
    model = DnCNN(depth=17, n_channels=64, image_channels=1).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Function to denoise the image
def denoise_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        denoised_tensor = model(image_tensor)
        denoised_img = denoised_tensor.squeeze().cpu().numpy()
        denoised_img = np.clip(denoised_img, 0, 1)  # Keep in [0,1] range
        return (denoised_img * 255).astype(np.uint8)  # Convert back to 0-255
