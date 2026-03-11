import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms

from models.detect_tumor import TumorDetector
from models.model import UNet
from models.tumor_classifier import TumorClassifier
from models.gradcam import GradCAM


device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("Brain Tumor AI System")

st.write("Upload an MRI image to detect, segment, and classify brain tumors.")

# -----------------------------
# Load models
# -----------------------------

@st.cache_resource
def load_models():

    detector = TumorDetector().to(device)
    detector.load_state_dict(torch.load("models/tumor_detector.pth", map_location=device))
    detector.eval()

    segmenter = UNet().to(device)
    segmenter.load_state_dict(torch.load("models/brain_tumor_unet.pth", map_location=device))
    segmenter.eval()

    classifier = TumorClassifier().to(device)
    classifier.load_state_dict(torch.load("models/tumor_classifier.pth", map_location=device))
    classifier.eval()

    return detector, segmenter, classifier


detector, segmenter, classifier = load_models()

# -----------------------------
# Upload MRI
# -----------------------------

uploaded_file = st.file_uploader("Upload MRI Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes,0)

    st.image(img, caption="Uploaded MRI", width="stretch")

# -----------------------------
# Tumor Detection
# -----------------------------

    img_det = cv2.resize(img,(224,224))
    img_det = img_det/255.0

    img_det = np.stack([img_det,img_det,img_det],axis=0)

    tensor_det = torch.tensor(img_det).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = detector(tensor_det)

    pred = torch.argmax(output,1).item()

    if pred == 0:

        st.success("No tumor detected")

    else:

        st.error("Tumor detected")

# -----------------------------
# Segmentation
# -----------------------------

        img_seg = cv2.resize(img,(256,256))
        img_seg = img_seg/255.0

        img4 = np.stack([img_seg,img_seg,img_seg,img_seg],axis=0)

        tensor_seg = torch.tensor(img4).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_mask = segmenter(tensor_seg)

        mask = pred_mask.squeeze().cpu().numpy()

        if len(mask.shape)==3:
            mask = np.argmax(mask,axis=0)

        mask = (mask>0).astype(np.uint8)

        tumor_pixels = np.sum(mask)
        total_pixels = mask.size

        tumor_percentage = (tumor_pixels/total_pixels)*100

        st.write("Tumor affected area:", round(tumor_percentage,2), "%")

# -----------------------------
# Overlay visualization
# -----------------------------

        mask_color = np.zeros((256,256,3),dtype=np.uint8)
        mask_color[mask>0] = [0,0,255]

        mri_rgb = cv2.cvtColor((img_seg*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(mri_rgb,0.7,mask_color,0.3,0)

        st.image(overlay, caption="Tumor Segmentation")

# -----------------------------
# Tumor Classification
# -----------------------------

        img_cls = cv2.resize(img,(224,224))
        img_cls = img_cls/255.0

        img_cls = np.stack([img_cls,img_cls,img_cls],axis=0)

        tensor_cls = torch.tensor(img_cls).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = classifier(tensor_cls)

        pred = torch.argmax(output,1).item()

        classes = ["Glioma","Meningioma","Pituitary"]

        tumor_type = classes[pred]

        st.subheader("Tumor Type Detected:")
        st.success(tumor_type)


# -----------------------------
            # Grad-CAM Visualization    
            # create GradCAM
        gradcam = GradCAM(classifier.model, classifier.model.layer4[1].conv2)

        # generate heatmap
        cam = gradcam.generate(tensor_cls)

        # check if GradCAM worked
        if cam is None:

            st.warning("Grad-CAM could not be generated.")

        else:

            # normalize heatmap
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # convert MRI to RGB
            mri_rgb = cv2.cvtColor(cv2.resize(img,(224,224)), cv2.COLOR_GRAY2RGB)

            # create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # overlay heatmap on MRI
            overlay = cv2.addWeighted(mri_rgb, 0.6, heatmap, 0.4, 0)

            st.image(overlay, caption="Grad-CAM Overlay (Model Attention)")