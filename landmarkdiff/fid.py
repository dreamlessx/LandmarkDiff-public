"""Self-contained FID computation using InceptionV3 feature extraction.

Avoids dependency on torch-fidelity by implementing FID directly.
Supports GPU acceleration, batched processing, and caching.

Usage:
    from landmarkdiff.fid import compute_fid_from_dirs, compute_fid_from_arrays

    # From directories
    fid = compute_fid_from_dirs("path/to/real", "path/to/generated")

    # From numpy arrays
    fid = compute_fid_from_arrays(real_images, generated_images)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _load_inception_v3():
    """Load InceptionV3 with pool3 features (2048-dim)."""
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # We want features from the avg pool layer (2048-dim)
    # Remove the final FC layer
    model.fc = nn.Identity()
    model.eval()
    return model


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a directory."""

    def __init__(self, directory: str | Path, image_size: int = 299):
        self.directory = Path(directory)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.files = sorted(
            f for f in self.directory.iterdir() if f.suffix.lower() in exts and f.is_file()
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import cv2

        img = cv2.imread(str(self.files[idx]))
        if img is None:
            # Return zeros if image can't be loaded
            return torch.zeros(3, self.image_size, self.image_size)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] then ImageNet normalize
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = _imagenet_normalize(t)
        return t


class NumpyArrayDataset(Dataset):
    """Dataset wrapping a list of numpy arrays."""

    def __init__(self, images: list[np.ndarray], image_size: int = 299):
        self.images = images
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import cv2

        img = self.images[idx]
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size))
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = _imagenet_normalize(t)
        return t


def _imagenet_normalize(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def _extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract InceptionV3 pool3 features from a dataloader."""
    features = []
    for batch in dataloader:
        batch = batch.to(device)
        feat = model(batch)
        if isinstance(feat, tuple):
            feat = feat[0]
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def _compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of feature vectors."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _calculate_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Calculate FID given two sets of statistics.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_from_dirs(
    real_dir: str | Path,
    generated_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
) -> float:
    """Compute FID between two directories of images.

    Args:
        real_dir: Path to real images.
        generated_dir: Path to generated images.
        batch_size: Batch size for feature extraction.
        num_workers: DataLoader workers.
        device: "cuda" or "cpu". Auto-detects if None.

    Returns:
        FID score (lower = better).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for FID computation")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = _load_inception_v3().to(dev)

    real_ds = ImageFolderDataset(real_dir)
    gen_ds = ImageFolderDataset(generated_dir)

    if len(real_ds) == 0 or len(gen_ds) == 0:
        raise ValueError("Need at least 1 image in each directory")

    real_loader = DataLoader(
        real_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    gen_loader = DataLoader(gen_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    real_features = _extract_features(model, real_loader, dev)
    gen_features = _extract_features(model, gen_loader, dev)

    mu_real, sigma_real = _compute_statistics(real_features)
    mu_gen, sigma_gen = _compute_statistics(gen_features)

    return _calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_fid_from_arrays(
    real_images: list[np.ndarray],
    generated_images: list[np.ndarray],
    batch_size: int = 32,
    device: str | None = None,
) -> float:
    """Compute FID from lists of numpy arrays.

    Args:
        real_images: List of (H, W, 3) BGR uint8 images.
        generated_images: List of (H, W, 3) BGR uint8 images.
        batch_size: Batch size for feature extraction.
        device: "cuda" or "cpu".

    Returns:
        FID score (lower = better).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for FID computation")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = _load_inception_v3().to(dev)

    real_ds = NumpyArrayDataset(real_images)
    gen_ds = NumpyArrayDataset(generated_images)

    real_loader = DataLoader(real_ds, batch_size=batch_size, num_workers=0)
    gen_loader = DataLoader(gen_ds, batch_size=batch_size, num_workers=0)

    real_features = _extract_features(model, real_loader, dev)
    gen_features = _extract_features(model, gen_loader, dev)

    mu_real, sigma_real = _compute_statistics(real_features)
    mu_gen, sigma_gen = _compute_statistics(gen_features)

    return _calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
