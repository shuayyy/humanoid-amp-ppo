from __future__ import annotations

import torch
from torch import nn

from defm.model_factory import create_defm_model
from defm.utils.utils import preprocess_depth_batch


class DeFMEncoder(nn.Module):
    """Frozen pretrained DeFM-B0 depth encoder."""

    output_dim = 192
    image_height = 224
    image_width = 224

    def __init__(self, device: torch.device | str) -> None:
        super().__init__()

        self.model = create_defm_model(
            "defm_efficientnet_b0",
            pretrained=True,
        ).to(device)

        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> DeFMEncoder:
        """Keep the frozen DeFM backbone in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: Metric depth from MJLab with shape [B, 224, 224, 1].

        Returns:
            Frozen DeFM global feature with shape [B, 192].
        """
        expected_shape = (
            self.image_height,
            self.image_width,
            1,
        )
        if depth.ndim != 4 or tuple(depth.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected depth shape [B, 224, 224, 1], "
                f"got {tuple(depth.shape)}."
            )

        if not torch.isfinite(depth).all():
            raise ValueError("Depth contains NaN or Inf.")

        # MJLab: [B, H, W, 1] -> DeFM: [B, 1, H, W].
        depth_bchw = depth.permute(0, 3, 1, 2).contiguous().float()

        # Official DeFM preprocessing: [B, 1, H, W] -> [B, 3, H, W].
        defm_input = preprocess_depth_batch(
            depth_bchw,
            target_size=(self.image_height, self.image_width),
            cnn_padding=True,
            device=depth.device,
        )

        output = self.model(defm_input)
        features = output["global_backbone"]

        expected_feature_shape = (depth.shape[0], self.output_dim)
        if tuple(features.shape) != expected_feature_shape:
            raise RuntimeError(
                f"Expected DeFM features {expected_feature_shape}, "
                f"got {tuple(features.shape)}."
            )

        return features
