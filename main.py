"""Load the HAM10000 dataset with PyTorch and create stratified data splits.

Expected layout in the project root:

data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
    ISIC_....jpg
  HAM10000_images_part_2/
    ISIC_....jpg
"""

# Neccessary imports
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Model 1 imports
from model1 import run_model1_pipeline


def build_transform(image_size: int = 224) -> transforms.Compose:
    """Create the image transform pipeline used by the dataset."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


class HAM10000Dataset(Dataset):
    """PyTorch dataset for HAM10000 images and labels."""

    def __init__(self, data_dir: str | Path = "data", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.metadata = self._load_metadata(self.data_dir)
        self.class_names = sorted(self.metadata["dx"].unique())
        self.class_to_idx = {label: index for index, label in enumerate(self.class_names)}
        self.metadata["label"] = self.metadata["dx"].map(self.class_to_idx)

    @staticmethod
    def _load_metadata(data_dir: Path) -> pd.DataFrame:
        metadata_csv = data_dir / "HAM10000_metadata.csv"
        part_1 = data_dir / "HAM10000_images_part_1"
        part_2 = data_dir / "HAM10000_images_part_2"

        missing_items = [
            str(path) for path in (metadata_csv, part_1, part_2) if not path.exists()
        ]
        if missing_items:
            missing_text = "\n - ".join([""] + missing_items)
            raise FileNotFoundError(
                "Missing required HAM10000 files/folders:" + missing_text
            )

        metadata_df = pd.read_csv(metadata_csv)
        required_columns = {"image_id", "dx"}
        missing_columns = required_columns.difference(metadata_df.columns)
        if missing_columns:
            raise ValueError(
                "Missing required columns in HAM10000_metadata.csv: "
                + ", ".join(sorted(missing_columns))
            )

        image_files = list(part_1.glob("*.jpg")) + list(part_2.glob("*.jpg"))
        image_map = {image.stem: str(image.resolve()) for image in image_files}
        metadata_df["image_path"] = metadata_df["image_id"].map(image_map)

        missing_paths = metadata_df["image_path"].isna()
        if missing_paths.any():
            missing_count = int(missing_paths.sum())
            raise FileNotFoundError(
                f"Found {missing_count} metadata rows without a matching image file."
            )

        return metadata_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_data_splits(
    dataset: HAM10000Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create stratified train/test subsets with preserved class proportions."""
    indices = dataset.metadata.index.to_numpy()
    labels = dataset.metadata["label"].to_numpy()

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_dataset = Subset(dataset, train_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())

    return train_dataset, test_dataset


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    test_size: float = 0.2,
    image_size: int = 224,
    random_state: int = 42,
):
    """Build dataset, stratified splits, and batch dataloaders."""
    transform = build_transform(image_size=image_size)
    dataset = HAM10000Dataset(data_dir=data_dir, transform=transform)
    train_dataset, test_dataset = create_data_splits(
        dataset,
        test_size=test_size,
        random_state=random_state,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":

    dataset, train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=64,
        test_size=0.2,
        image_size=224,
        random_state=42,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {dataset.class_names}")
    print(f"Classes in training set: {train_dataset.dataset.class_names}")
    print(f"Classes in testing set: {test_dataset.dataset.class_names}")

    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch tensor shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    # train_dataset.dataset.metadata.head().to_csv("metadata_head.csv", index=False) # Uncomment to save a sample of the metadata

    # Run full ResNet50 feature extraction + SVM pipeline.
    run_model1_pipeline(train_loader, train_dataset, test_dataset)