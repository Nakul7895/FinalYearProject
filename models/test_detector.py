import torch
import cv2
import numpy as np
from detect_tumor import TumorDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TumorDetector().to(device)

model.load_state_dict(torch.load("models/tumor_detector.pth", map_location=device))
model.eval()

# test image
img = cv2.imread("test_images/tumor_test.png")

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img,(224,224))
img = img/255.0

img = np.transpose(img,(2,0,1))

tensor = torch.tensor(img).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(tensor)

pred = torch.argmax(output,1).item()

if pred == 0:
    print("No tumor detected")
else:
    print("Tumor detected")