import optuna
from optuna.pruners import SuccessiveHalvingPruner
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import yaml

from src.datasets.seeg_dataset import SEEGDataset
from src.models.model import SEEGFusionModel
from src.training.train import get_subject_indices, compute_class_weights, move_to_device
from src.utils import load_yaml


def suggest_from_cfg(trial, name, param_cfg):
    ptype = param_cfg["type"]
    if ptype == "int":
        if "step" in param_cfg:
            return trial.suggest_int(
                name, param_cfg["low"], param_cfg["high"], step=param_cfg["step"]
            )
        return trial.suggest_int(name, param_cfg["low"], param_cfg["high"])
    if ptype == "categorical":
        # supports int or str choices
        return trial.suggest_categorical(name, param_cfg["choices"])
    if ptype == "loguniform":
        return trial.suggest_float(name, param_cfg["low"], param_cfg["high"], log=True)
    if ptype == "uniform":
        return trial.suggest_float(name, param_cfg["low"], param_cfg["high"])
    raise ValueError(f"Unknown param type: {ptype} for {name}")


def objective(trial, tune_cfg, device):
    proxy_subjects = tune_cfg["subjects"]
    n_folds = tune_cfg["n_folds"]

    # Trial hyperparameters
    search = tune_cfg["search"]  # your search spaces map

    # Dynamically suggest all hyperparameters from tune_cfg
    trial_params = {}
    for hp_name, hp_cfg in search.items():
        trial_params[hp_name] = suggest_from_cfg(trial, hp_name, hp_cfg)

    trial.set_user_attr("trial_params", trial_params)

    # Extract hyperparameters
    embed_dim = trial_params["embed_dim"]
    num_layers = trial_params["num_layers"]
    n_heads = trial_params["n_heads"]
    weight_decay = trial_params["weight_decay"]
    base_lr = trial_params["base_lr"]
    max_lr = trial_params["max_lr"]
    pct_start = trial_params["pct_start"]

    # Fixed tuning parameters
    batch_size = tune_cfg["batch_size"]
    n_epochs = tune_cfg["n_epochs"]
    patience = tune_cfg["patience"]

    fold_metrics = []

    # Load dataset restricted to proxy subjects
    full_dataset = SEEGDataset(subjects=tune_cfg["subjects"], embed_dim=embed_dim, verbose=False)

    # K-fold proxy tuning: treat each proxy subject as val once
    for val_subj in proxy_subjects[:n_folds]:
        train_subjs = [s for s in proxy_subjects if s != val_subj]

        train_idx = get_subject_indices(full_dataset, train_subjs)
        val_idx = get_subject_indices(full_dataset, [val_subj])

        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        }

        # Model
        model = SEEGFusionModel(
            embed_dim=embed_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            n_classes=2,
            device=device,
        )
        model.to(device)

        # Class weights
        weights = compute_class_weights(train_ds)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=n_epochs,
            steps_per_epoch=len(dataloaders["train"]),
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        # TRAIN WITH PRUNING
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            # ---- Train one epoch ----
            model.train()
            for inputs, labels in dataloaders["train"]:
                inputs = move_to_device(inputs, device)
                labels = move_to_device(labels, device)

                optimizer.zero_grad()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

            # ---- Validation ----
            model.eval()
            val_loss = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in dataloaders["val"]:
                    inputs = move_to_device(inputs, device)
                    labels = move_to_device(labels, device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs["convergent"].size(0)

            epoch_val_loss = val_loss / len(dataloaders["val"].dataset)

            # Report epoch metric to Optuna and enable pruning
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early stopping inside each fold
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        fold_metrics.append(best_val_loss)

    # The objective returns average validation loss across folds
    return float(np.mean(fold_metrics))


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tune_cfg yaml
    tune_cfg = load_yaml("src/config/tune.yaml")

    study = optuna.create_study(
        direction="minimize",
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            bootstrap_count=1,
        ),
    )

    study.optimize(
        lambda trial: objective(trial, tune_cfg, device),
        n_trials=50,
        timeout=None,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    best = study.best_trial
    with open("../experiments/best_trial.yaml", "w") as f:
        yaml.safe_dump(best.user_attrs["trial_params"], f)

    print("Best trial:", study.best_trial)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":

    main()
