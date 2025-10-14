from __future__ import annotations

import os 

from torch.utils.data import random_split

from mmpfn.datasets.pad_ufes_20 import PADUFES20Dataset

import os 
import torch 
import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score
from mmpfn.models.mmpfn import MMPFNClassifier
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

import optuna


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data_path = os.path.join(os.getenv('HOME'), "workspace/research/MultiModalPFN/mmpfn/data/pad_ufes_20")
data_path = os.path.join(os.getenv('HOME'), "works/research/MultiModalPFN/mmpfn/data/pad_ufes_20")
dataset = PADUFES20Dataset(data_path)
# _ = dataset.get_images()
_ = dataset.get_embeddings()


def objective(trial):
    
    mgm_heads = trial.suggest_categorical("mgm_heads", [2, 4, 8, 16, 32, 64, 128])
    cap_heads = trial.suggest_categorical("cap_heads", [2, 4, 8, 12])

    accuracy_scores = []
    for seed in range(5):
        torch.manual_seed(seed)
        train_len = int(len(dataset) * 0.8)
        test_len = len(dataset) - train_len

        train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

        X_train = train_dataset.dataset.x[train_dataset.indices]
        y_train = train_dataset.dataset.y[train_dataset.indices]
        X_test = test_dataset.dataset.x[test_dataset.indices]
        y_test = test_dataset.dataset.y[test_dataset.indices]
        image_train = train_dataset.dataset.embeddings[train_dataset.indices]
        image_test = test_dataset.dataset.embeddings[test_dataset.indices]
        
        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1
        for i in range(X_test.shape[1]):
            col = X_test[:, i]
            col[np.isnan(col)] = np.nanmin(col) - 1

        torch.cuda.empty_cache()

        save_path_to_fine_tuned_model = "./checkpoints/finetuned_mmpfn_pad_ufes_20.ckpt"
        
        fine_tune_mmpfn(
            # path_to_base_model="auto",
            save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
            # Finetuning HPs
            time_limit=60,
            finetuning_config={"learning_rate": 0.00001, "batch_size": 1, "max_steps": 100},
            validation_metric="log_loss",
            # Input Data
            X_train=pd.DataFrame(X_train),
            image_train=image_train,
            y_train=pd.Series(y_train),
            categorical_features_index=None,
            device="cuda",  # use "cpu" if you don't have a GPU
            task_type="multiclass",
            # Optional
            show_training_curve=False,  # Shows a final report after finetuning.
            logger_level=0,  # Shows all logs, higher values shows less
            freeze_input=True,  # Freeze the input layers (encoder and y_encoder) during finetuning
            mixer_type='MGM+CAP', # MGM MGM+CAP
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
        )

        # disables preprocessing at inference time to match fine-tuning
        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name='none')]
        )

        # Evaluate on Test Data
        model_finetuned = MMPFNClassifier(
            model_path=save_path_to_fine_tuned_model,
            inference_config=no_preprocessing_inference_config, 
            ignore_pretraining_limits=True,
            mixer_type='MGM+CAP', # MGM MGM+CAP
            mgm_heads=mgm_heads,
            cap_heads=cap_heads,
        )

        clf_finetuned = model_finetuned.fit(X_train, image_train, y_train)
        acc_score = accuracy_score(y_test, clf_finetuned.predict(X_test, image_test))
        print("accuracy_score (Finetuned):", acc_score)
        accuracy_scores.append(acc_score)

    # get mean and std of accuracy scores
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print("Mean Accuracy:", mean_accuracy)
    print("Std Accuracy:", std_accuracy)
    
    return mean_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print results
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)