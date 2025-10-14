from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from mmpfn.scripts_finetune_mm.metric_utils.ag_metrics import Scorer
    from mmpfn.models.mmpfn.model.transformer import PerFeatureTransformer


def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    image_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_samples: int,
    is_classification: bool,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
]:
    # Split data ourselves
    if n_samples < 10000:
        test_size = 0.2#0.33
    elif n_samples < 500000:
        test_size = 0.2
    elif n_samples < 1000000:
        test_size = 0.1
    else:
        test_size = 0.05
        
    if image_train is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=rng,
            stratify=y_train if is_classification else None,
        )
        return X_train, X_val, None, None, y_train, y_val
    elif X_train is None:
        image_train, image_val, y_train, y_val = train_test_split(
            image_train,
            y_train,
            test_size=test_size,
            random_state=rng,
            stratify=y_train if is_classification else None,
        )
        return None, None, image_train, image_val, y_train, y_val    
    X_train, X_val, image_train, image_val, y_train, y_val = train_test_split(
        X_train,
        image_train,
        y_train,
        test_size=test_size,
        random_state=rng,
        stratify=y_train if is_classification else None,
    )
    return X_train, X_val, image_train, image_val, y_train, y_val


def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    image_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,  # (n_samples, batch_size, 1)
    image_val: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_val: torch.Tensor,  # (n_samples, batch_size, 1)
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
) -> float:
    """Validate the TabPFN model and return a loss (lower is better).

    This code assumes that batch_size for validation is 1. Otherwise,
    need to write a loop, I guess?
    """
    if X_train is not None:
        X_train = X_train.to(device)
        X_val = X_val.to(device)
    
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    
    if image_train is not None:
        image_train = image_train.to(device)
        image_val = image_val.to(device)
    
    pred_logits = model_forward_fn(
        model=model,
        X_train=X_train,
        image_train=image_train,
        y_train=y_train,
        X_test=X_val,
        image_test=image_val,
        forward_for_validation=True,
    )

    match task_type:
        case TaskType.REGRESSION:
            y_pred = pred_logits.float().flatten().cpu().detach().numpy()
            y_true = y_val.float().flatten().cpu().detach().numpy()
        case TaskType.BINARY_CLASSIFICATION:
            # TODO: check that this works / is exhaustive.
            if validation_metric.needs_threshold or validation_metric.needs_proba:
                y_pred = (
                    torch.nn.functional.sigmoid(pred_logits[:, 0, 1])
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                # Required to get the correct classes for the metrics
                y_pred = (
                    torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
            y_true = y_val.long().flatten().cpu().detach().numpy()
        case TaskType.MULTICLASS_CLASSIFICATION:
            y_pred = (
                torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
            y_true = y_val.long().flatten().cpu().detach().numpy()
        case _:
            raise ValueError(f"Task type {task_type} not supported.")

    score = validation_metric(y_true=y_true, y_pred=y_pred)

    if X_train is not None:
        X_train.cpu()
        X_val.cpu()
    
    y_train.cpu()
    y_val.cpu()
    
    if image_train is not None:
        image_train.cpu()
        image_val.cpu()
    
    return validation_metric.convert_score_to_error(score=score)
