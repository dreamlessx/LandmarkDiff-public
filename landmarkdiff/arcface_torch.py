"""PyTorch-native ArcFace model for differentiable identity loss.

Drop-in replacement for the ONNX-based InsightFace ArcFace used in losses.py.
The original IdentityLoss extracts embeddings under @torch.no_grad(), which
means the identity loss term contributes zero gradients during Phase B training.
This module provides a fully differentiable path so that gradients flow back
through the predicted image into the ControlNet.

Architecture: IResNet-50 matching the InsightFace w600k_r50 ONNX model.
  conv1(3->64, 3x3, bias) -> PReLU ->
  4 IResNet stages [3, 4, 14, 3] with channels [64, 128, 256, 512] ->
  BN2d -> Flatten -> FC(512*7*7 -> 512) -> BN1d -> L2-normalize

Each IBasicBlock: BN -> conv3x3(bias) -> PReLU -> conv3x3(bias) + residual.
No SE module. Convolutions use bias=True.

Pretrained weights: converted from the InsightFace buffalo_l w600k_r50.onnx
model to a PyTorch state dict (backbone.pth). The conversion extracts weights
from the ONNX graph and maps them to matching PyTorch module keys.

Usage in losses.py:
    from landmarkdiff.arcface_torch import ArcFaceLoss
    identity_loss = ArcFaceLoss(device=device)
    loss = identity_loss(pred_image, target_image)  # gradients flow through pred
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class IBasicBlock(nn.Module):
    """Improved basic residual block for IResNet.

    Structure: BN -> conv3x3(bias) -> PReLU -> conv3x3(bias) -> + residual
    Uses pre-activation style BatchNorm. Convolutions have bias=True to match
    the InsightFace w600k_r50 ONNX weights.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ArcFaceBackbone(nn.Module):
    """IResNet-50 backbone for ArcFace identity embeddings.

    Input:  (B, 3, 112, 112) face crops normalized to [-1, 1].
    Output: (B, 512) L2-normalized embeddings.

    Architecture matches the InsightFace w600k_r50 ONNX model exactly:
    Conv(bias) -> PReLU -> 4 stages -> BN2d -> Flatten -> FC -> BN1d -> L2norm.
    """

    def __init__(
        self,
        layers: tuple[int, ...] = (3, 4, 14, 3),
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.inplanes = 64

        # Stem: conv1(bias) -> PReLU (no BN in stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(64)

        # 4 residual stages
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Head: BN2d -> Flatten -> FC -> BN1d
        self.bn2 = nn.BatchNorm2d(512, eps=2e-5, momentum=0.1)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=2e-5, momentum=0.1)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Conv2d(
                self.inplanes,
                planes,
                kernel_size=1,
                stride=stride,
                bias=True,
            )

        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(IBasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) in [-1, 1].

        Returns:
            (B, 512) L2-normalized embeddings.
        """
        x = self.conv1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

# Known locations where converted backbone.pth may live
_KNOWN_WEIGHT_PATHS = [
    Path.home() / ".cache" / "arcface" / "backbone.pth",
    Path.home() / ".insightface" / "models" / "buffalo_l" / "backbone.pth",
]


def _find_pretrained_weights() -> Path | None:
    """Search known locations for pretrained IResNet-50 weights."""
    for p in _KNOWN_WEIGHT_PATHS:
        if p.exists() and p.suffix == ".pth" and p.stat().st_size > 0:
            return p
    return None


def load_pretrained_weights(
    model: ArcFaceBackbone,
    weights_path: str | None = None,
) -> bool:
    """Load pretrained InsightFace IResNet-50 weights into the model.

    Weights are a PyTorch state dict converted from the InsightFace
    w600k_r50.onnx model. Key names match our module structure exactly.

    Args:
        model: An ``ArcFaceBackbone`` instance.
        weights_path: Explicit path to a ``.pth`` file.  If ``None``, searches
            known locations.

    Returns:
        ``True`` if weights were loaded successfully, ``False`` otherwise
        (model keeps random initialization).
    """
    path: Path | None = None

    if weights_path is not None:
        path = Path(weights_path)
        if not path.exists():
            logger.warning("Specified weights path does not exist: %s", path)
            path = None

    if path is None:
        path = _find_pretrained_weights()

    if path is None:
        warnings.warn(
            "No pretrained ArcFace weights found. The model will use random "
            "initialization. Identity loss values will be meaningless until "
            "proper weights are loaded. Place backbone.pth at "
            f"{Path.home() / '.cache' / 'arcface' / 'backbone.pth'}",
            UserWarning,
            stacklevel=2,
        )
        return False

    logger.info("Loading ArcFace weights from %s", path)
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)

    # Handle the case where the checkpoint wraps the state dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Try direct load first
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded ArcFace weights (strict match)")
        return True
    except RuntimeError:
        pass

    # Try non-strict load (some checkpoints may have extra keys)
    try:
        # Remap common differences
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("output_layer."):
                new_k = k.replace("output_layer.", "features.")
            remapped[new_k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning(
                "Missing keys when loading ArcFace weights: %s",
                missing[:10],
            )
        if unexpected:
            logger.info("Unexpected keys (ignored): %s", unexpected[:10])
        logger.info("Loaded ArcFace weights (non-strict)")
        return True
    except Exception as e:
        warnings.warn(
            f"Failed to load ArcFace weights from {path}: {e}. Using random initialization.",
            UserWarning,
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# Differentiable face alignment
# ---------------------------------------------------------------------------


def align_face(
    images: torch.Tensor,
    size: int = 112,
) -> torch.Tensor:
    """Center-crop and resize face images to (size x size) differentiably.

    Uses ``F.grid_sample`` with bilinear interpolation so that gradients
    flow back through the spatial transform into the input images.

    The crop extracts the central 80% of the image (removes background
    padding that is common in generated 512x512 face images) and resizes
    to the target size.

    Args:
        images: (B, 3, H, W) tensor, any normalization.
        size: Target spatial size (default 112 for ArcFace).

    Returns:
        (B, 3, size, size) tensor with the same normalization as input.
    """
    B, C, H, W = images.shape

    if H == size and W == size:
        return images

    # Crop fraction: keep central 80% to remove background padding
    crop_frac = 0.8

    # Build a normalized grid [-1, 1] covering the center crop region
    half_crop = crop_frac / 2.0
    theta = torch.zeros(B, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = half_crop  # x scale
    theta[:, 1, 1] = half_crop  # y scale

    grid = F.affine_grid(theta, [B, C, size, size], align_corners=False)
    aligned = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return aligned


def align_face_no_crop(
    images: torch.Tensor,
    size: int = 112,
) -> torch.Tensor:
    """Resize face images to (size x size) without cropping, differentiably.

    Simple bilinear resize using ``F.interpolate`` for gradient flow. Use
    this when images are already tightly cropped faces.

    Args:
        images: (B, 3, H, W) tensor.
        size: Target spatial size.

    Returns:
        (B, 3, size, size) tensor.
    """
    if images.shape[-2] == size and images.shape[-1] == size:
        return images
    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


# ---------------------------------------------------------------------------
# ArcFaceLoss: differentiable identity preservation loss
# ---------------------------------------------------------------------------


class ArcFaceLoss(nn.Module):
    """Differentiable identity loss using PyTorch-native ArcFace.

    Replaces the ONNX-based InsightFace ArcFace in ``IdentityLoss`` from
    ``losses.py``. Gradients flow through the predicted image into the
    generator, while the target embedding is detached.

    Loss = mean(1 - cosine_similarity(embed(pred), embed(target).detach()))

    The backbone is frozen (no gradient updates to ArcFace itself) but
    gradients DO flow through the forward pass of the backbone when
    computing pred embeddings.

    Example::

        loss_fn = ArcFaceLoss(device=torch.device("cuda"))
        loss = loss_fn(pred_images, target_images)
        loss.backward()  # gradients flow into pred_images
    """

    def __init__(
        self,
        device: torch.device | None = None,
        weights_path: str | None = None,
        crop_face: bool = True,
    ):
        """
        Args:
            device: Device to place the backbone on. If ``None``, determined
                from the first forward call.
            weights_path: Path to pretrained backbone.pth. If ``None``,
                searches known locations.
            crop_face: Whether to center-crop images before embedding.
                Set ``False`` if images are already 112x112 face crops.
        """
        super().__init__()
        self.crop_face = crop_face
        self._weights_path = weights_path
        self._target_device = device
        self._initialized = False

        # Build backbone (lazy device placement)
        self.backbone = ArcFaceBackbone()

    def _ensure_initialized(self, device: torch.device) -> None:
        """Lazy initialization: load weights and move to device on first use."""
        if self._initialized:
            return

        # Load pretrained weights
        loaded = load_pretrained_weights(self.backbone, self._weights_path)
        if not loaded:
            logger.warning(
                "ArcFaceLoss using random weights -- identity loss will not "
                "be meaningful. Download pretrained weights for proper training."
            )

        # Move to device and freeze
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self._initialized = True

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Prepare images for ArcFace: crop, resize, normalize to [-1, 1].

        Args:
            images: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 3, 112, 112) in [-1, 1].
        """
        x = align_face(images, size=112) if self.crop_face else align_face_no_crop(images, size=112)

        # Normalize from [0, 1] to [-1, 1]
        x = x * 2.0 - 1.0
        return x

    def _extract_embedding(
        self,
        images: torch.Tensor,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """Extract ArcFace embeddings.

        Args:
            images: (B, 3, 112, 112) in [-1, 1].
            enable_grad: If ``True``, gradients flow through the backbone's
                forward pass (used for pred). If ``False``, detached (target).

        Returns:
            (B, 512) L2-normalized embeddings.
        """
        if enable_grad:
            return self.backbone(images)
        else:
            with torch.no_grad():
                return self.backbone(images)

    def forward(
        self,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        procedure: str = "rhinoplasty",
    ) -> torch.Tensor:
        """Compute differentiable identity loss.

        Args:
            pred_image: (B, 3, H, W) predicted images in [0, 1].
                Gradients WILL flow back through this tensor.
            target_image: (B, 3, H, W) target images in [0, 1].
                Gradients will NOT flow through this (detached).
            procedure: Surgical procedure type. ``"orthognathic"`` returns
                zero loss (identity irrelevant for jaw surgery).

        Returns:
            Scalar loss: mean(1 - cosine_similarity(pred_emb, target_emb)).
            Returns 0 for orthognathic or empty batches.
        """
        if procedure == "orthognathic":
            return torch.tensor(0.0, device=pred_image.device, dtype=pred_image.dtype)

        device = pred_image.device
        self._ensure_initialized(device)

        # Procedure-specific cropping (before ArcFace alignment)
        pred_crop = self._procedure_crop(pred_image, procedure)
        target_crop = self._procedure_crop(target_image, procedure)

        # Prepare for ArcFace (crop, resize to 112x112, normalize to [-1, 1])
        pred_prepared = self._prepare_images(pred_crop)
        target_prepared = self._prepare_images(target_crop)

        # Extract embeddings
        pred_emb = self._extract_embedding(pred_prepared, enable_grad=True)
        target_emb = self._extract_embedding(target_prepared, enable_grad=False)

        # Detach target to be absolutely sure no gradients leak
        target_emb = target_emb.detach()

        # Cosine similarity loss: 1 - cos_sim
        cosine_sim = (pred_emb * target_emb).sum(dim=1)  # (B,)

        # Clamp to valid range (numerical safety for BF16)
        cosine_sim = cosine_sim.clamp(-1.0, 1.0)

        loss = (1.0 - cosine_sim).mean()
        return loss

    def _procedure_crop(
        self,
        image: torch.Tensor,
        procedure: str,
    ) -> torch.Tensor:
        """Crop image based on surgical procedure for identity comparison."""
        _, _, h, w = image.shape

        if procedure == "rhinoplasty":
            return image[:, :, : h * 2 // 3, :]
        elif procedure == "blepharoplasty":
            return image
        elif procedure == "rhytidectomy":
            return image[:, :, : h * 3 // 4, :]
        else:
            return image

    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Extract identity embeddings (utility method for evaluation).

        Args:
            images: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 512) L2-normalized embeddings (detached).
        """
        self._ensure_initialized(images.device)
        prepared = self._prepare_images(images)
        return self._extract_embedding(prepared, enable_grad=False)


# ---------------------------------------------------------------------------
# Convenience: create a pre-configured loss instance
# ---------------------------------------------------------------------------


def create_arcface_loss(
    device: torch.device | None = None,
    weights_path: str | None = None,
) -> ArcFaceLoss:
    """Factory function for creating an ArcFaceLoss with sensible defaults.

    Args:
        device: Target device (auto-detected if ``None``).
        weights_path: Path to backbone.pth (auto-searched if ``None``).

    Returns:
        Configured ``ArcFaceLoss`` instance.
    """
    return ArcFaceLoss(device=device, weights_path=weights_path)
