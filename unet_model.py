import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Contracting Path
        self.conv1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = conv_block(512, 1024)

        # Expansive Path
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = conv_block(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = conv_block(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = conv_block(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = conv_block(128, 64)

        self.output = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)
        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)
        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)
        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = self.output(c9)
        return self.sigmoid(outputs)

# Load the trained U-Net model
def load_unet_model(model_path):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess the noisy image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Generate a denoised image using U-Net
def denoise_image_unet(model, image):
    with torch.no_grad():
        denoised_img = model(image).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    return (denoised_img * 255).astype(np.uint8)
