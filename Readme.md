# 🧠 BrainTumorAI – Final Year Project (Comprehensive Documentation)

> A full-stack deep learning system for **brain MRI analysis**. It
> performs automatic tumour detection, segmentation, classification, and
> visualization, with a Streamlit front-end and PyTorch back-end. This
> document explains algorithms, models, data, architecture, and usage in
detail.

---

## 📌 Project Overview

BrainTumorAI is designed as an intuitive tool for medical
professionals and researchers to analyze brain MRI scans.
The system integrates several deep learning components into a single
pipeline:

1. **Tumor Detection** – A binary classifier determines whether a scan
   contains any tumour.
2. **Segmentation** – If a tumour is present, a U-Net model produces a
   pixel-wise mask showing the affected region.
3. **Classification** – The segmented tumour is classified into one of
   three types: Glioma, Meningioma, or Pituitary.
4. **Interpretability** – Grad-CAM generates a heatmap indicating areas
   the classifier focused on.
5. **Reporting** – PDF summaries are generated for record keeping.

A React-like Streamlit web UI allows users to upload MRIs and view
results interactively. Model weights are stored locally so the entire
project can be cloned and run without retraining.

---

## 🔍 Algorithms & Models

### 1. Tumor Detection
- **Architecture:** ResNet-18 (pretrained on ImageNet).
- **Modifications:** Feature extractor frozen to prevent overfitting; a
  custom classifier head (512→128→ReLU→Dropout→2) predicts tumour
  presence.
- **Training:** Supervised learning using cross-entropy loss and Adam
  optimizer. Data augmentation (resize, horizontal flip, rotation)
  increases robustness.
- **Usage:** models/TumorDetector in models/detect_tumor.py.

### 2. Tumor Segmentation
- **Architecture:** Simplified 2-level U-Net implemented in
  models/model.py.
- **Channels:** Five convolutional layers including down‑path and up‑path
  with skip connections. Input has 4 identical grayscale channels to
  match training format.
- **Output:** 3-channel mask (background + two tumour classes). Output
  passed through a sigmoid to produce probabilities.
- **Training Data:** Paired MRI images and segmentation masks (not
  included), typically produced with manual annotation or synthetic
  methods.

### 3. Tumor Classification
- **Architecture:** ResNet-18 again, but fully trainable (no frozen
  layers) with a 3-class head (512→128→ReLU→Dropout→3).
- **Classes:** Glioma, Meningioma, Pituitary. Softmax applied during
  inference.
- **Training:** Similar augmentation pipeline as detection. Cross-entropy
  loss, Adam optimizer, 10 epochs by default.

### 4. Grad‑CAM
- **Purpose:** Provide visual explanations of classifier decisions.
- **Implementation:** models/gradcam.py grabs activations from the final
  convolutional layer (layer4[1].conv2) of ResNet-18 and computes a
  weighted sum of gradients to produce a heatmap.

### 5. Reporting
- **Generation:** `report_generator.py` uses ReportLab to create a PDF
  showing tumour type and area percentage.

---

## 🗂 Datasets

### Classification Dataset
- **Source:** Kaggle  Brain Tumor MRI Dataset by masoudnickparvar.
- **Structure:** dataset_classification/ with three folders, one per
  tumour type. Each contains MRI scans in .jpg/.png format.
- **Acquisition:** download_classifier_dataset.py downloads and copies
  the data into the project using the kagglehub helper.

### Detection Dataset
- **Structure:** dataset_detection/no/ (healthy) and
  dataset_detection/yes/ (tumour present).
- **Preparation:** Created manually or derived from classification dataset
  by grouping scans by whether they contain tumours.

### Segmentation Data
- **Not included** due to size/annotation complexity. Users are expected
  to supply pairings of images and masks in a similar folder structure.

> 💡 **Tip:** For segmentation training, you can follow the U‑Net
> tutorial on PyTorch’s website or use tools like LabelBox/Roboflow to
> produce masks.

---

## 🛠 Tech Stack

| Layer        | Tools / Libraries                 | Purpose                         |
|--------------|-----------------------------------|---------------------------------|
| Front‑end    | Streamlit                         | Web interface, file uploads     |
| Back‑end     | PyTorch, torchvision              | Model definition & inference    |
| Image I/O    | OpenCV (opencv-python)          | Loading, resizing, overlaying   |
| Data         | NumPy                            | Array manipulations             |
| Reporting    | ReportLab                        | Generate PDF diagnosis reports  |
| Dataset mgmt | `torchvision.datasets.ImageFolder` | Dataset loading for training    |
| CLI tools    | Standard Python (os, shutil, etc.)| Downloading & utilities         |

The application runs entirely on Python 3.8+ and can leverage CUDA if
available.

---

## 🔄 Workflow & Architecture

1. **User uploads an MRI** via Streamlit.
2. The image is converted to grayscale and normalized.
3. **Detection model** processes a 224×224 version to decide presence.
   - If no tumour: result shown and pipeline stops.
4. If tumour present:
   - **Segmentation model** runs on a 256×256 4-channel tensor. The
     mask is thresholded, optionally cleaned with morphology, and
     percentage area computed.
   - **Visualization:** Red overlay of mask on MRI is displayed.
   - **Classification model** takes a 224×224 version and predicts type.
   - **Grad‑CAM** produces a heatmap using classifier gradients; overlay
     displayed to indicate attention.
   - **Report generation** is optionally triggered to produce a PDF.

All models are loaded once at startup (cached by @st.cache_resource).
This reduces latency during interactive sessions. The directory
structure supports separate training scripts that save weights under
models/ for later use.

A simplified architectural diagram is provided below:

`mermaid
flowchart LR
    U[Upload MRI] --> D[Detector (ResNet-18)]
    D -- Tumour? No --> S1[Stop / Message]
    D -- Tumour? Yes --> S2[Segmentation (UNet)]
    S2 --> V[Overlay Visualization]
    S2 --> C[Classifier (ResNet-18)]
    C --> G[Grad-CAM]
    G --> V2[Attention Overlay]
    V --> U2[Show Results]
    V2 --> U2
` 

(For presentation, consider drawing a more detailed architecture with
block diagrams of each model.)

---

## 🗃 Project Structure (Scanned)

`
BrainTumorAI/
├── app.py                     # Streamlit user interface, inference flow
├── download_classifier_dataset.py  # Kaggle downloader script
├── report_generator.py        # PDF report helper
├── requirements.txt           # Core dependencies
├── .gitignore                 # Ignore rules (keeps models committed)
├── classification/            # Training/prediction utilities for classifier
│   ├── train_classifier.py    # Training script for tumour type classifier
│   └── predict_classifier.py  # CLI inference for classifier
├── dataset_classification/    # Downloaded classification images
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/
├── dataset_detection/         # Images separated into yes/no tumour
│   ├── yes/
│   └── no/
├── models/                    # Model definitions & training scripts
│   ├── detect_tumor.py        # TumourDetector class
│   ├── tumor_classifier.py    # TumorClassifier class
│   ├── model.py               # UNet definition
│   ├── gradcam.py             # Grad-CAM helper
│   ├── train_detector.py      # Detector training script
│   ├── train_classifier.py    # Classifier training script
│   ├── brain_tumor_unet.pth   # trained segmentation weights
│   ├── tumor_detector.pth     # trained detection weights
│   ├── tumor_classifier.pth   # trained classification weights
│   └── ...                    # other weight files as required
├── segmentation/              # Standalone segmentation prediction
│   └── predict.py
├── test_images/               # Example images for quick testing
├── outputs/                   # Generated results, reports, masks
└── Readme.md                  # This documentation file
`

> 🔎 *Note:* The weight files are intentionally committed so the full
> project can be cloned and run without retraining.

---

## 📦 Dependencies & Installation

See earlier installation steps; the 
equirements.txt contains core
packages. Additional tools (e.g. kagglehub) are installed
separately if used.

`ash
pip install -r requirements.txt
pip install streamlit reportlab kagglehub
`

Optional: install CUDA and configure PyTorch for GPU acceleration.

---

## 🧪 Evaluation & Results (Suggested Metrics)

- **Detection:** Accuracy, sensitivity, specificity, confusion matrix.
- **Segmentation:** Dice score, Intersection over Union (IoU), pixel
  accuracy.
- **Classification:** Accuracy per class, ROC curves, precision / recall.
- **Grad‑CAM:** Qualitative assessment – are highlighted areas
  medically relevant?

Graph results and show example images when presenting.

---

## 📊 Front‑End / Back‑End Tech Notes

- **Front‑end**: Streamlit handles layout, widgets, and state. When a
  file is uploaded, callbacks trigger model inference synchronously.
- **Back‑end**: PyTorch models loaded with 	orch.load(...). Inference
  is wrapped in with torch.no_grad() for speed.
- **Data flow**: Raw bytes → NumPy → torch Tensor → model → NumPy →
  OpenCV for display.

Networking, database, or API layers are not part of this prototype but
could be added for deployment.

---

## 🎓 Presentation Advice

1. Introduce the medical problem and motivations.
2. Describe data collection and preprocessing.
3. Explain each model component and why it was chosen.
4. Demonstrate the system live using pp.py.
5. Show quantitative results and qualitative visualizations.
6. Discuss limitations, ethical considerations, and future work.
7. Provide GitHub link and README for reviewers.

Include architecture diagrams, data samples, and annotated screenshots in
your slides.

---

## 🎯 Next Steps & Extensions

- Add segmentation training script and include annotated masks.
- Extend models to handle 3D MRI volumes (e.g., using 3D U‑Net).
- Deploy as a web service with Docker or a cloud platform.
- Integrate additional interpretability methods (e.g., integrated
  gradients).
- Build a proper database to log predictions and user feedback.

---

This comprehensive README should serve both as a project report and a
user/developer guide. Feel free to modify it further to suit your
presentation needs. Good luck with your submission! 🚀
