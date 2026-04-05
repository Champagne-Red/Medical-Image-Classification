"""Model 1: ResNet50 feature extraction + SVM classifier.

This script reuses the dataset/dataloader pipeline from main.py.
"""

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models

def build_resnet50_feature_extractor(device: torch.device) -> torch.nn.Module:

    """Load pretrained ResNet50 and remove final classification layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor


def extract_cnn_features(dataloader, feature_extractor: torch.nn.Module, device: torch.device):
    
    """Forward all images from dataloader and return flattened CNN features + labels."""
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = feature_extractor(images)  # [batch, 2048, 1, 1]
            outputs = outputs.view(outputs.size(0), -1)  # [batch, 2048]

            all_features.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

    return features, labels


def train_svm(features: np.ndarray, labels: np.ndarray):
    
    """Train a simple SVM on extracted CNN features."""
    svm_model = make_pipeline(StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
    )
    svm_model.fit(features, labels)

    return svm_model


def run_model1_pipeline(train_loader, train_dataset, test_dataset):
    
    """Run ResNet50 feature extraction on train split and fit an SVM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = build_resnet50_feature_extractor(device)

    train_features, train_labels = extract_cnn_features(
    train_loader,
    feature_extractor,
    device,
    )
    svm_model = train_svm(train_features, train_labels)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Extracted train feature shape: {train_features.shape}")
    print(f"Extracted train label shape: {train_labels.shape}")
    print("SVM training finished.")

    return svm_model