# 🧠 BrainTumorAI – Final Year Project

> A complete deep‑learning system for **brain MRI analysis**:
> detecting the presence of a tumour, segmenting the affected area,  
> classifying the tumour type, and visualising model attention with Grad‑CAM.  
> Built with PyTorch and deployed as a Streamlit web app.

---

## 📌 Project Overview

This repository contains the code, data‑handling scripts, trained models, and documentation
for a final‑year project that demonstrates end‑to‑end medical image
processing using convolutional neural networks.  
The workflow supported by the system is:

1. **Detection** – binary classification (tumour / no‑tumour)  
2. **Segmentation** – pixel‑wise tumour mask using a U‑Net  
3. **Classification** – multi‑class tumour type (Glioma, Meningioma, Pituitary)  
4. **Interpretability** – Grad‑CAM heatmap highlighting regions of interest  
5. **Reporting** – PDF summary of results  

A Streamlit interface (`app.py`) allows the user to upload an MRI scan and see
all stages executed sequentially.

---

## 🏗 Architecture & Models

| Stage       | Model class           | Architecture / Notes                                   | File(s)                       |
|-------------|-----------------------|--------------------------------------------------------|-------------------------------|
| Detector    | `TumorDetector`       | ResNet‑18 backbone (pre‑trained) with frozen features;<br>custom 2‑layer classifier for 2 classes | `models/detect_tumor.py`      |
| Segmenter   | `UNet`                | Simple 2‑level U‑Net; 4‑channel input; 3‑class output   | `models/model.py`             |
| Classifier  | `TumorClassifier`     | ResNet‑18 backbone (pre‑trained); fine‑tuned fully‑connected head with 3 outputs | `models/tumor_classifier.py`  |
| Grad‑CAM    | `GradCAM`             | Helper for generating attention maps from classifier   | `models/gradcam.py`           |

Each model is trained on the corresponding folder under `dataset_detection` or
`dataset_classification` (see next section). Trained weights are stored in
`models/*.pth` and loaded by the app.

---

## 📂 Dataset Structure

- **Classification** (`dataset_classification/`)  
  - `glioma/`  
  - `meningioma/`  
  - `pituitary/`  
  Images are organised for `torchvision.datasets.ImageFolder`.

- **Detection** (`dataset_detection/`)  
  - `yes/` – scans containing a tumour  
  - `no/` – healthy scans

- **Segmentation**  
  - Not included in repo; assumed to be prepared elsewhere as binary masks
    or synthesised as part of development.

A helper script (`download_classifier_dataset.py`) pulls the
[Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset).

---

## ⚙️ Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone <repo-url>
   cd BrainTumorAI
   ```

2. **Create a virtual environment** (recommended)  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   pip install streamlit reportlab kagglehub  # if you use the downloader/report
   ```

4. **Prepare datasets**  
   - Run `python download_classifier_dataset.py` to obtain the classification data.
   - Populate `dataset_detection` manually or via your own script.
   - (Optional) Add segmentation masks and modify `models/model.py` training accordingly.

5. **Train models** *(optional if you just want to use pretrained weights)*  
   ```bash
   python models/train_detector.py
   python models/train_classifier.py
   # segmentation training script not provided – implement as needed
   ```

   Trained weights will be saved under `models/` as `.pth` files.

---

## 🚀 Running the Application

Start the Streamlit UI:

```bash
streamlit run app.py
```

Open the link shown in your browser, upload an MRI image (`.png`, `.jpg`, `.jpeg`),
and watch the pipeline execute:

1. detection → message
2. segmentation → numeric tumour area & overlay image
3. classification → type label
4. Grad‑CAM overlay for interpretability
5. (Optional) generate a PDF report using `report_generator.create_report`

Example command‑line scripts for each component are available in
`classification/predict_classifier.py` and `segmentation/predict.py`.

---

## 📊 Evaluation & Results

For demonstration and presentation, you should report:

- **Detection accuracy** and confusion matrix on held‑out samples.
- **Classifier accuracy** per tumour type, precision / recall.
- **Segmentation metrics**: Dice coefficient / IoU (compute using your masks).
- **Visual examples**: original MRI, predicted mask, Grad‑CAM heatmap.
- **Runtime performance** on CPU/GPU (inference times).

Include screenshots from the Streamlit app and PDF reports in your slides.

---

## 🎯 Final Year Project Presentation Tips

1. **Start with motivation** – why automatic tumour analysis matters.
2. **Explain data sources** and preprocessing steps.
3. **Walk through the 3‑stage pipeline** with architecture diagrams.
4. **Live demo** – run the Streamlit app on a sample image.
5. **Discuss challenges**: limited data, segmentation annotation,
   model generalisation.
6. **Highlight interpretability** via Grad‑CAM and reporting.
7. **Mention future work**: 3D volumes, more classes, explainable AI, deployment.

---

## 🗂 Project Structure Overview

```
.
├── app.py                     # Streamlit front-end
├── download_classifier_dataset.py
├── report_generator.py
├── requirements.txt
├── classification/            # training & prediction helpers
├── dataset_classification/
├── dataset_detection/
├── models/                    # architectures and training scripts
│   ├── detect_tumor.py
│   ├── model.py
│   ├── tumor_classifier.py
│   ├── train_detector.py
│   ├── train_classifier.py
│   ├── gradcam.py
│   └── *.pth                 # trained weights
├── segmentation/              # standalone prediction script
├── test_images/               # sample MRIs
└── outputs/                   # results, reports, etc.
```

---

## ✅ Dependencies

- Python 3.8+
- torch, torchvision
- opencv-python, numpy, matplotlib
- streamlit (for UI)
- reportlab (for PDF report)
- kagglehub (dataset downloader, optional)

---

## 📌 Notes & Credits

- Models are adapted from standard ResNet‑18 and U‑Net architectures.
- Dataset courtesy of Kaggle user *masoudnickparvar*.
- This repository is intended as a **complete final year project submission**;
  feel free to extend it with more data, models, or deployment strategies.

---

Good luck with your presentation! Let me know if you need help adding
visuals or writing report sections 🙌
