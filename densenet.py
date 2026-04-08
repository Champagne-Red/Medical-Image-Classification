"""Model 2: DenseNet121 fine-tuning for HAM10000.

This module is designed to be imported from the notebook and used with the
train/test dataloaders created in main.ipynb.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn
from torchvision import models


def build_densenet121(num_classes: int, device: torch.device) -> nn.Module:
	"""Load pretrained DenseNet121 and replace the classifier head."""
	model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
	in_features = model.classifier.in_features
	model.classifier = nn.Linear(in_features, num_classes)
	model = model.to(device)

	return model


def build_weighted_cross_entropy(train_labels, device: torch.device) -> nn.Module:
	"""Create weighted cross entropy from train-label class frequencies."""
	labels = np.asarray(train_labels, dtype=np.int64)
	class_counts = Counter(labels.tolist())
	num_classes = int(labels.max()) + 1

	# Inverse-frequency weights with safe fallback for missing classes.
	weights = []
	total = float(len(labels))
	for class_idx in range(num_classes):
		count = class_counts.get(class_idx, 0)
		if count == 0:
			weights.append(0.0)
		else:
			weights.append(total / (num_classes * float(count)))

	weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
	criterion = nn.CrossEntropyLoss(weight=weight_tensor)

	return criterion


def train_densenet121(
	model: nn.Module,
	train_loader,
	criterion: nn.Module,
	device: torch.device,
	epochs: int = 7,
	learning_rate: float = 1e-4,
):
	"""Train DenseNet121 for a fixed number of epochs (default: 7)."""
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	history = []

	model.train()
	for epoch in range(epochs):
		running_loss = 0.0
		total_samples = 0

		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			batch_size = labels.size(0)
			running_loss += float(loss.item()) * batch_size
			total_samples += batch_size

		epoch_loss = running_loss / max(total_samples, 1)
		history.append(epoch_loss)
		print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

	return model, history


def predict_densenet(model: nn.Module, dataloader, device: torch.device):
	"""Run inference and return numpy arrays of predictions and labels."""
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(device)
			logits = model(images)
			preds = torch.argmax(logits, dim=1)

			all_preds.append(preds.cpu().numpy())
			all_labels.append(labels.numpy())

	predictions = np.concatenate(all_preds, axis=0)
	labels = np.concatenate(all_labels, axis=0)

	return predictions, labels


def run_densenet_pipeline(train_loader, train_dataset, train_labels, epochs: int = 7):
	"""Train DenseNet121 using the provided train loader/dataset.

	Note: The train transform is expected to already be applied by train_loader,
	which matches the create_dataloaders logic in the notebook.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	num_classes = int(np.asarray(train_labels, dtype=np.int64).max()) + 1

	model = build_densenet121(num_classes=num_classes, device=device)
	criterion = build_weighted_cross_entropy(train_labels=train_labels, device=device)

	model, history = train_densenet121(
		model=model,
		train_loader=train_loader,
		criterion=criterion,
		device=device,
		epochs=epochs,
	)

	print(f"Device: {device}")
	print(f"Train samples: {len(train_dataset)}")
	print(f"Epochs: {epochs}")
	print("DenseNet121 training finished.")

	return model, history

