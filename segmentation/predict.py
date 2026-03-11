import torch
import cv2
import numpy as np
from models.model import UNet

# Select device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = UNet().to(device)
model.load_state_dict(torch.load("models/brain_tumor_unet.pth", map_location=device))
model.eval()

# Load MRI image
img = cv2.imread("test_images/tumor_test.png", 0)

if img is None:
    print("Image not found. Check the path.")
    exit()

# Resize and normalize
img = cv2.resize(img, (256, 256))
img = img / 255.0

# Model expects 4 MRI channels
img4 = np.stack([img, img, img, img], axis=0)

tensor = torch.tensor(img4).unsqueeze(0).float().to(device)

# Run prediction
with torch.no_grad():
    pred = model(tensor)

# Convert prediction to numpy
mask = pred.squeeze().cpu().numpy()

mask = (mask > 0.5).astype(np.uint8)

kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Convert to class labels
mask_class = np.argmax(mask, axis=0)

# Save segmentation mask
mask_img = (mask_class * 85).astype(np.uint8)
cv2.imwrite("predicted_mask.png", mask_img)

print("Prediction completed. Saved as predicted_mask.png")

# Tumor statistics
tumor_pixels = np.sum(mask_class > 0)
total_pixels = mask_class.size
tumor_percentage = (tumor_pixels / total_pixels) * 100

print("Tumor pixels:", tumor_pixels)
print("Tumor percentage:", tumor_percentage)

# Show detected classes
print("Classes detected:", np.unique(mask_class))

# Create colored tumor mask
mask_color = np.zeros((256, 256, 3), dtype=np.uint8)

# Highlight tumor in red
mask_color[mask_class > 0] = [0, 0, 255]

# Convert MRI to RGB
mri_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Overlay tumor mask on MRI
overlay = cv2.addWeighted(mri_rgb, 0.7, mask_color, 0.3, 0)

# Save overlay result
cv2.imwrite("tumor_overlay.png", overlay)

print("Overlay saved as tumor_overlay.png")