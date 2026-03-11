import streamlit as st
import torch
import cv2
import numpy as np

from models.detect_tumor import TumorDetector
from models.model import UNet
from models.tumor_classifier import TumorClassifier
from models.gradcam import GradCAM

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Custom Dark UI
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#0b0b0b;
    color:white;
}

.hero{
    text-align:center;
    padding-top:100px;
    padding-bottom:50px;
}

.hero-title{
    font-size:55px;
    font-weight:700;
}

.hero-sub{
    font-size:22px;
    margin-top:10px;
    color:#cfcfcf;
}

.hero-desc{
    margin-top:10px;
    font-size:17px;
    color:#9e9e9e;
}

.section-title{
    font-size:26px;
    font-weight:600;
    margin-top:40px;
}

.result-box{
    background:#141414;
    padding:20px;
    border-radius:10px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""
<div class="hero">

<div class="hero-title">
🧠 Brain Tumor AI System
</div>

<div class="hero-sub">
Automated MRI Tumor Detection & Analysis
</div>

<div class="hero-desc">
Upload an MRI scan to detect, segment and classify brain tumors using deep learning.
</div>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load Models
# --------------------------------------------------

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

# --------------------------------------------------
# Upload MRI
# --------------------------------------------------

st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png","jpg","jpeg"]
)

tumor_present = False
tumor_percentage = None

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes,0)

# --------------------------------------------------
# Tumor Detection
# --------------------------------------------------

    img_det = cv2.resize(img,(224,224))
    img_det = img_det/255.0

    img_det = np.stack([img_det,img_det,img_det],axis=0)

    tensor_det = torch.tensor(img_det).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = detector(tensor_det)

    pred = torch.argmax(output,1).item()

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if pred == 0:
        st.success("No tumor detected")

    else:
        tumor_present = True
        st.error("Tumor detected")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Segmentation + Classification + GradCAM
# --------------------------------------------------

    if tumor_present:

        # Segmentation
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

        # Overlay segmentation
        mask_color = np.zeros((256,256,3),dtype=np.uint8)
        mask_color[mask>0] = [0,0,255]

        mri_rgb = cv2.cvtColor((img_seg*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

        segmentation_img = cv2.addWeighted(mri_rgb,0.7,mask_color,0.3,0)

        # Classification
        img_cls = cv2.resize(img,(224,224))
        img_cls = img_cls/255.0

        img_cls = np.stack([img_cls,img_cls,img_cls],axis=0)

        tensor_cls = torch.tensor(img_cls).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = classifier(tensor_cls)

        pred = torch.argmax(output,1).item()

        classes = ["Glioma","Meningioma","Pituitary"]

        tumor_type = classes[pred]

        # GradCAM
        gradcam = GradCAM(classifier.model, classifier.model.layer4[1].conv2)

        cam = gradcam.generate(tensor_cls)

        if cam is not None:

            cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)

            heatmap = cv2.applyColorMap(
                np.uint8(255*cam),
                cv2.COLORMAP_JET
            )

            heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

            mri_rgb2 = cv2.cvtColor(
                cv2.resize(img,(224,224)),
                cv2.COLOR_GRAY2RGB
            )

            gradcam_img = cv2.addWeighted(
                mri_rgb2,
                0.7,
                heatmap,
                0.3,
                0
            )

# --------------------------------------------------
# Display Results (SIDE BY SIDE)
# --------------------------------------------------

        st.markdown('<div class="section-title">MRI Analysis Results</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img, caption="MRI Scan", width=300)

        with col2:
            st.image(segmentation_img, caption="Tumor Segmentation", width=300)

        with col3:
            st.image(gradcam_img, caption="Grad-CAM", width=300)

# --------------------------------------------------
# Tumor Stats
# --------------------------------------------------

                
        st.markdown("## 🧠 Tumor Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Tumor Type Detected",
                value=tumor_type
            )

        with col2:
            st.metric(
                label="Tumor Affected Area",
                value=f"{round(tumor_percentage,2)} %"
            )
# --------------------------------------------------
# Tumor Growth Prediction (Improved)
# --------------------------------------------------

if tumor_present:

    st.markdown('<div class="section-title">📈 Tumor Growth Prediction</div>', unsafe_allow_html=True)

    st.write("Upload MRI scans in chronological order (Month 1 → Month 2 → Month 3).")

    # Reset button
    if st.button("Reset Uploaded Scans"):
        st.session_state.pop("growth_upload", None)

    growth_files = st.file_uploader(
        "Upload MRI scans",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True,
        key="growth_upload"
    )

    if growth_files:

        tumor_sizes = []

        # Process in upload order (no sorting)
        for file in growth_files:

            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes,0)

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

            tumor_sizes.append(tumor_percentage)

        # ----------------------------------------
        # Show sizes per scan
        # ----------------------------------------

        st.markdown("### Tumor Size per Scan")

        for i, size in enumerate(tumor_sizes):
            st.write(f"Month {i+1}: {round(size,2)} %")

        # ----------------------------------------
        # Growth calculation
        # ----------------------------------------

        if len(tumor_sizes) >= 2:

            growth_rates = []

            for i in range(1, len(tumor_sizes)):
                growth_rates.append(tumor_sizes[i] - tumor_sizes[i-1])

            avg_growth = sum(growth_rates) / len(growth_rates)

            latest_size = tumor_sizes[-1]
            predicted_size = latest_size + avg_growth

            # ----------------------------------------
            # Metrics
            # ----------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Latest Tumor Size",
                    value=f"{round(latest_size,2)} %"
                )

            with col2:
                st.metric(
                    label="Average Growth Rate",
                    value=f"{round(avg_growth,2)} %"
                )

            with col3:
                st.metric(
                    label="Predicted Next Size",
                    value=f"{round(predicted_size,2)} %"
                )

            # ----------------------------------------
            # Prediction color logic
            # ----------------------------------------

            if predicted_size > latest_size:
                color = "red"
                message = "Tumor size is predicted to increase"

            elif abs(predicted_size - latest_size) < 0.1:
                color = "yellow"
                message = "Tumor size is predicted to remain stable"

            else:
                color = "green"
                message = "Tumor size is predicted to decrease"

            # ----------------------------------------
            # Colored prediction result
            # ----------------------------------------

            st.markdown(
                f"<h3 style='color:{color};'>Predicted tumor size for next scan: {round(predicted_size,2)} %</h3>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<h4 style='color:{color};'>{message}</h4>",
                unsafe_allow_html=True
            )

        else:

            st.warning("Please upload at least 2 scans to calculate tumor growth.")