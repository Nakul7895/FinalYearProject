import kagglehub
import shutil
import os

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Dataset downloaded to:", path)

source = os.path.join(path, "Training")

destination = "dataset_classification"

shutil.copytree(source, destination, dirs_exist_ok=True)

print("Dataset copied to dataset_classification folder")