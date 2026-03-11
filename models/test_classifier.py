import torch
import cv2
import numpy as np
from tumor_classifier import TumorClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TumorClassifier().to(device)
model.load_state_dict(torch.load("models/tumor_classifier.pth"))
model.eval()

img = cv2.imread("test_images/tumor_test.png")

img = cv2.resize(img,(224,224))
img = img/255.0

img = np.transpose(img,(2,0,1))

tensor = torch.tensor(img).unsqueeze(0).float().to(device)

with torch.no_grad():

    output = model(tensor)

pred = torch.argmax(output,1).item()

classes = ["Glioma","Meningioma","Pituitary"]

print("Tumor Type:",classes[pred])