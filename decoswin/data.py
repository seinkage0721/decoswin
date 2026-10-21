"""Dataset adapters: users supply images and a split manifest, not dataset code."""

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


def prepare_image(image, size=128):
    """White-pad without changing aspect ratio, then resize to the paper's input size."""
    image = image.convert("RGB")
    side = max(image.size)
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    image = ImageOps.expand(image, (left, top, side - image.width - left, side - image.height - top), fill="white")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def image_transform(train=False):
    # This intentionally keeps augmentation modest. No horizontal flip: it can
    # change a glyph's identity. Extra corruption in the paper is omitted.
    augment = [transforms.RandomRotation(15, fill=(255, 255, 255)), transforms.ColorJitter(brightness=0.15, contrast=0.15)] if train else []
    return transforms.Compose(augment + [transforms.ToTensor()])


def read_manifest(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,split columns")
        rows = []
        for line, row in enumerate(reader, 2):
            if not row["path"] or not row["label"] or not row["split"]:
                raise ValueError(f"Missing field on manifest line {line}")
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            rows.append((image_path, row["label"].strip(), row["split"].strip().lower()))
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


class ManifestDataset(Dataset):
    def __init__(self, rows, split, labels, train=False):
        self.items = [(p, labels[label]) for p, label, s in rows if s == split and label in labels]
        unknown = {label for _, label, s in rows if s == split and label not in labels}
        if unknown:
            raise ValueError(f"{split} has labels absent from training: {sorted(unknown)[:5]}")
        if not self.items:
            raise ValueError(f"No images found for split '{split}'")
        self.transform = image_transform(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            return self.transform(prepare_image(image)), label


def training_labels(rows):
    labels = sorted({label for _, label, split in rows if split == "train"})
    if not labels:
        raise ValueError("Manifest needs a train split")
    return {label: i for i, label in enumerate(labels)}


def class_counts(dataset, num_classes):
    counts = Counter(label for _, label in dataset.items)
    return [counts[i] for i in range(num_classes)]


class UnlabeledImages(Dataset):
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return TF.to_tensor(prepare_image(image))
