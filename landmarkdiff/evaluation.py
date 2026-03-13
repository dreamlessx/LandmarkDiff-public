"""Evaluation metrics: FID, LPIPS, NME, ArcFace sim, SSIM.

Stratified by Fitzpatrick skin type (I-VI) via ITA thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


@dataclass
class EvalMetrics:
    """Computed evaluation metrics for a batch of generated images."""

    fid: float = 0.0
    lpips: float = 0.0
    nme: float = 0.0           # Normalized Mean landmark Error
    identity_sim: float = 0.0  # ArcFace cosine similarity
    ssim: float = 0.0

    # Per-Fitzpatrick breakdown (all metrics stratified)
    fid_by_fitzpatrick: dict[str, float] = field(default_factory=dict)
    nme_by_fitzpatrick: dict[str, float] = field(default_factory=dict)
    lpips_by_fitzpatrick: dict[str, float] = field(default_factory=dict)
    ssim_by_fitzpatrick: dict[str, float] = field(default_factory=dict)
    identity_sim_by_fitzpatrick: dict[str, float] = field(default_factory=dict)
    count_by_fitzpatrick: dict[str, int] = field(default_factory=dict)

    # Per-procedure breakdown
    nme_by_procedure: dict[str, float] = field(default_factory=dict)
    lpips_by_procedure: dict[str, float] = field(default_factory=dict)
    ssim_by_procedure: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"FID:          {self.fid:.2f}",
            f"LPIPS:        {self.lpips:.4f}",
            f"NME:          {self.nme:.4f}",
            f"Identity Sim: {self.identity_sim:.4f}",
            f"SSIM:         {self.ssim:.4f}",
        ]
        if self.count_by_fitzpatrick:
            lines.append("\nBy Fitzpatrick Type:")
            for ftype in sorted(self.count_by_fitzpatrick):
                n = self.count_by_fitzpatrick[ftype]
                parts = [f"  Type {ftype} (n={n}):"]
                if ftype in self.lpips_by_fitzpatrick:
                    parts.append(f"LPIPS={self.lpips_by_fitzpatrick[ftype]:.4f}")
                if ftype in self.ssim_by_fitzpatrick:
                    parts.append(f"SSIM={self.ssim_by_fitzpatrick[ftype]:.4f}")
                if ftype in self.nme_by_fitzpatrick:
                    parts.append(f"NME={self.nme_by_fitzpatrick[ftype]:.4f}")
                if ftype in self.identity_sim_by_fitzpatrick:
                    parts.append(f"ID={self.identity_sim_by_fitzpatrick[ftype]:.4f}")
                lines.append(" ".join(parts))
        if self.fid_by_fitzpatrick:
            lines.append("\nFID by Fitzpatrick:")
            for k, v in sorted(self.fid_by_fitzpatrick.items()):
                lines.append(f"  Type {k}: {v:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to flat dictionary for JSON/CSV export."""
        d = {
            "fid": self.fid,
            "lpips": self.lpips,
            "nme": self.nme,
            "identity_sim": self.identity_sim,
            "ssim": self.ssim,
        }
        for ftype in sorted(self.count_by_fitzpatrick):
            prefix = f"fitz_{ftype}"
            d[f"{prefix}_count"] = self.count_by_fitzpatrick.get(ftype, 0)
            d[f"{prefix}_lpips"] = self.lpips_by_fitzpatrick.get(ftype, 0.0)
            d[f"{prefix}_ssim"] = self.ssim_by_fitzpatrick.get(ftype, 0.0)
            d[f"{prefix}_nme"] = self.nme_by_fitzpatrick.get(ftype, 0.0)
            d[f"{prefix}_identity"] = self.identity_sim_by_fitzpatrick.get(ftype, 0.0)
        for proc in sorted(self.nme_by_procedure):
            d[f"proc_{proc}_nme"] = self.nme_by_procedure.get(proc, 0.0)
            d[f"proc_{proc}_lpips"] = self.lpips_by_procedure.get(proc, 0.0)
            d[f"proc_{proc}_ssim"] = self.ssim_by_procedure.get(proc, 0.0)
        return d


def classify_fitzpatrick_ita(image: np.ndarray) -> str:
    """Fitzpatrick I-VI from ITA angle (Chardon et al. 1991 thresholds)."""
    if cv2 is None:
        raise ImportError("opencv-python is required for Fitzpatrick classification")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample from face center region (avoid background)
    h, w = image.shape[:2]
    center = lab[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    L_mean = center[:, :, 0].mean() * 100 / 255  # scale to 0-100
    b_mean = center[:, :, 2].mean() - 128  # center around 0

    if abs(b_mean) < 1e-6:
        b_mean = 1e-6

    ita = np.arctan2(L_mean - 50, b_mean) * (180 / np.pi)

    if ita > 55:
        return "I"
    elif ita > 41:
        return "II"
    elif ita > 28:
        return "III"
    elif ita > 10:
        return "IV"
    elif ita > -30:
        return "V"
    else:
        return "VI"


def compute_nme(
    pred_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    left_eye_idx: int = 33,
    right_eye_idx: int = 263,
) -> float:
    """Compute Normalized Mean Error for landmarks."""
    iod = np.linalg.norm(
        target_landmarks[left_eye_idx] - target_landmarks[right_eye_idx]
    )
    if iod < 1.0:
        iod = 1.0

    distances = np.linalg.norm(pred_landmarks - target_landmarks, axis=1)
    return float(np.mean(distances) / iod)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """SSIM via skimage, falls back to global SSIM if not installed."""
    try:
        from skimage.metrics import structural_similarity
        # Convert to grayscale if color, or compute per-channel
        if pred.ndim == 3 and pred.shape[2] == 3:
            return float(structural_similarity(pred, target, channel_axis=2, data_range=255))
        else:
            return float(structural_similarity(pred, target, data_range=255))
    except ImportError:
        # Fallback: simple global SSIM (not publication-quality)
        pred_f = pred.astype(np.float64)
        target_f = target.astype(np.float64)

        mu_p = np.mean(pred_f)
        mu_t = np.mean(target_f)
        sigma_p = np.std(pred_f)
        sigma_t = np.std(target_f)
        sigma_pt = np.mean((pred_f - mu_p) * (target_f - mu_t))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim_val = (
            (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        ) / (
            (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p ** 2 + sigma_t ** 2 + C2)
        )
        return float(ssim_val)


_LPIPS_FN = None


def _get_lpips_fn():
    """Get or create singleton LPIPS model."""
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False)
        _LPIPS_FN.eval()
    return _LPIPS_FN


def compute_lpips(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """LPIPS perceptual distance (lower = more similar)."""
    try:
        import lpips
        import torch
    except ImportError:
        return 0.0

    _lpips_fn = _get_lpips_fn()

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return t * 2 - 1  # LPIPS expects [-1, 1]

    with torch.no_grad():
        score = _lpips_fn(_to_tensor(pred), _to_tensor(target))
    return float(score.item())


def compute_fid(
    real_dir: str,
    generated_dir: str,
) -> float:
    """Compute FID between directories of real and generated images."""
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise ImportError(
            "torch-fidelity is required for FID. Install with: pip install torch-fidelity"
        )

    metrics = calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        cuda=True,
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def compute_identity_similarity(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """ArcFace cosine sim [0,1]. Falls back to SSIM if no InsightFace."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(112, 112))

        pred_bgr = pred if pred.shape[2] == 3 else cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        target_bgr = target if target.shape[2] == 3 else cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

        pred_faces = app.get(pred_bgr)
        target_faces = app.get(target_bgr)

        if pred_faces and target_faces:
            pred_emb = pred_faces[0].embedding
            target_emb = target_faces[0].embedding
            sim = np.dot(pred_emb, target_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            return float(np.clip(sim, 0, 1))
    except Exception:
        pass

    # Fallback: SSIM-based proxy
    return compute_ssim(pred, target)


def evaluate_batch(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    pred_landmarks: list[np.ndarray] | None = None,
    target_landmarks: list[np.ndarray] | None = None,
    procedures: list[str] | None = None,
    compute_identity: bool = False,
) -> EvalMetrics:
    """Evaluate a batch of predicted vs target images."""
    n = len(predictions)
    ssim_scores = []
    lpips_scores = []
    nme_scores = []
    identity_scores = []
    fitz_groups: dict[str, list[int]] = {}
    proc_groups: dict[str, list[int]] = {}

    for i in range(n):
        ssim_scores.append(compute_ssim(predictions[i], targets[i]))
        lpips_scores.append(compute_lpips(predictions[i], targets[i]))

        if pred_landmarks is not None and target_landmarks is not None:
            nme_scores.append(compute_nme(pred_landmarks[i], target_landmarks[i]))

        if compute_identity:
            identity_scores.append(compute_identity_similarity(predictions[i], targets[i]))

        # Fitzpatrick classification
        if cv2 is not None:
            try:
                fitz = classify_fitzpatrick_ita(targets[i])
                fitz_groups.setdefault(fitz, []).append(i)
            except Exception:
                pass

        # Procedure grouping
        if procedures is not None and i < len(procedures):
            proc_groups.setdefault(procedures[i], []).append(i)

    metrics = EvalMetrics(
        ssim=float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        lpips=float(np.mean(lpips_scores)) if lpips_scores else 0.0,
        nme=float(np.mean(nme_scores)) if nme_scores else 0.0,
        identity_sim=float(np.mean(identity_scores)) if identity_scores else 0.0,
    )

    # Full Fitzpatrick stratification for ALL metrics
    for ftype, indices in fitz_groups.items():
        metrics.count_by_fitzpatrick[ftype] = len(indices)

        group_lpips = [lpips_scores[i] for i in indices]
        if group_lpips:
            metrics.lpips_by_fitzpatrick[ftype] = float(np.mean(group_lpips))

        group_ssim = [ssim_scores[i] for i in indices]
        if group_ssim:
            metrics.ssim_by_fitzpatrick[ftype] = float(np.mean(group_ssim))

        if nme_scores:
            group_nme = [nme_scores[i] for i in indices if i < len(nme_scores)]
            if group_nme:
                metrics.nme_by_fitzpatrick[ftype] = float(np.mean(group_nme))

        if identity_scores:
            group_id = [identity_scores[i] for i in indices if i < len(identity_scores)]
            if group_id:
                metrics.identity_sim_by_fitzpatrick[ftype] = float(np.mean(group_id))

    # Per-procedure breakdown
    for proc, indices in proc_groups.items():
        group_lpips = [lpips_scores[i] for i in indices]
        if group_lpips:
            metrics.lpips_by_procedure[proc] = float(np.mean(group_lpips))

        group_ssim = [ssim_scores[i] for i in indices]
        if group_ssim:
            metrics.ssim_by_procedure[proc] = float(np.mean(group_ssim))

        if nme_scores:
            group_nme = [nme_scores[i] for i in indices if i < len(nme_scores)]
            if group_nme:
                metrics.nme_by_procedure[proc] = float(np.mean(group_nme))

    return metrics
