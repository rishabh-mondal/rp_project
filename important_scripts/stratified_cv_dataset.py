import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import shutil

# Define the path to your dataset
path = '/home/rishabh.mondal/rp_project/solar_panel_5'

# Define transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the ImageFolder dataset
dataset = datasets.ImageFolder(root=path, transform=transform)

# Define the number of splits for cross-validation
n_splits = 4  # 4-fold cross-validation

# Initialize stratified k-fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate over the folds and save them to separate directories
for fold_index, (train_indices, val_indices) in enumerate(stratified_kfold.split(dataset.imgs, dataset.targets)):
    fold_directory = f'/home/rishabh.mondal/rp_project/rp_project/crossval_data/solar_panel_4/fold_{fold_index + 1}'  # create directory names like 'fold_1', 'fold_2', ...
    os.makedirs(os.path.join(fold_directory, 'train'), exist_ok=True)  # create 'train' directory if it doesn't exist
    os.makedirs(os.path.join(fold_directory, 'val'), exist_ok=True)    # create 'val' directory if it doesn't exist

    # Copy images from dataset to fold directory for training
    for idx in train_indices:
        img_path, _ = dataset.imgs[idx]
        img_name = os.path.basename(img_path)
        class_name = dataset.classes[dataset.targets[idx]]
        target_dir = os.path.join(fold_directory, 'train', class_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(target_dir, img_name))

    # Copy images from dataset to fold directory for validation
    for idx in val_indices:
        img_path, _ = dataset.imgs[idx]
        img_name = os.path.basename(img_path)
        class_name = dataset.classes[dataset.targets[idx]]
        target_dir = os.path.join(fold_directory, 'val', class_name)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(target_dir, img_name))

    print(f"Fold {fold_index + 1} saved in '{fold_directory}/' directory.")

