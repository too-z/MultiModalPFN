from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from mmpfn.models.tabpfn_v2.model.transformer import PerFeatureTransformer


def save_model(
    *,
    model: PerFeatureTransformer,
    save_path_to_fine_tuned_model: Path,
    checkpoint_config: dict,
) -> None:
    """Save the fine-tuned model to disk in a TabPFN-readable checkpoint format."""
    # -- Save fine-tuned model
    torch.save(
        dict(state_dict=model.state_dict(), config=checkpoint_config),
        path=str(save_path_to_fine_tuned_model),
    )
