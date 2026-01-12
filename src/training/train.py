from datetime import datetime
import time
from tqdm import tqdm
import math
import random
from pathlib import Path
from loguru import logger
from argparse import ArgumentParser
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from src.utils import load_yaml, move_to_device
from src.datasets.seeg_dataset import SEEGDataset
from src.models.model import SEEGFusionModel, BaselineModel
from src.training.evaluate import evaluate_model


def sample_hyperparameters(kwargs, cohort_id, hp_id):
    """
    Sample a single hyperparameter configuration from the YAML-defined
    search space in kwargs['parameters']["search"].

    Args:
        kwargs (dict): loaded YAML config
        rng (np.random.Generator or None): optional RNG for reproducibility

    Returns:
        dict: sampled hyperparameters
    """

    rng = np.random.default_rng(SEED + hp_id)

    search_space = kwargs["parameters"]["search"]
    hp = {}

    for name, spec in search_space.items():
        hp_type = spec["type"]

        if hp_type == "int":
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step", 1)

            values = list(range(low, high + 1, step))
            hp[name] = int(rng.choice(values))

        elif hp_type == "categorical":
            hp[name] = int(rng.choice(spec["choices"]))

        elif hp_type == "uniform":
            hp[name] = float(rng.uniform(spec["low"], spec["high"]))

        elif hp_type == "loguniform":
            try:
                low = math.log(spec["low"])
            except KeyError:
                low = math.log(hp["base_lr"])
            high = math.log(spec["high"])
            hp[name] = float(math.exp(rng.uniform(low, high)))

        else:
            raise ValueError(f"Unknown hyperparameter type: {hp_type}")

    save_dir = Path(f"../../experiments/seed{SEED}/cohort{cohort_id}")
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / f"cohort{cohort_id}_hp{hp_id}.json", "w") as f:
        json.dump(hp, f, indent=2)

    return hp


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    save_prefix,
    cohort_id,
    scheduler=None,
    n_epochs=50,
    n_steps_per_update=1,
    patience=5,
    use_val=True,
    use_tensorboard=False,
    **kwargs,
):
    """
    Train a model with early stopping and per-epoch checkpoint saving.
    Args:
        model: nn.Module
        dataloaders: dict with 'train' and 'val' DataLoaders
        criterion: loss function
        optimizer: optimizer
        device: torch device
        save_prefix: prefix for saving out model weights
        cohort_id: cohort ID for saving
        scheduler: learning rate scheduler (optional)
        n_epochs: int, number of epochs (default: 50)
        n_steps_per_update: gradient accumulation steps (default: 1)
        patience: early stopping patience (# consecutive epochs without improvement) (default: 5)
        use_val: whether to use validation set for early stopping (default: True)
        use_tensorboard: whether to log to TensorBoard (default: False)

    Returns:
        model: trained model (with best weights loaded)
        history: dict with losses and accuracies
        best_epoch: epoch with best validation loss
    """
    save_dir = Path(f"../../experiments/seed{SEED}/cohort{cohort_id}")
    save_dir.mkdir(exist_ok=True, parents=True)
    model.to(device)

    # --- TensorBoard setup ---
    if use_tensorboard:
        tb_run_name = f"{save_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=f"{kwargs['paths']['tb_logs_path']}/{tb_run_name}")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"\nStarting training for {n_epochs} epochs on device: {device}")

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        train_correct = 0

        # --- TRAIN LOOP ---
        for step, (inputs, labels) in enumerate(
            tqdm(dataloaders["train"], desc=f"Epoch {epoch} [train]", leave=False)
        ):
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if torch.isnan(loss):
                logger.error(f"NaN loss detected at batch {step}")
                continue

            # global_step starts from 0
            global_step = (epoch - 1) * len(dataloaders["train"]) + step
            if use_tensorboard:
                writer.add_scalar("Batch/Loss", loss.item(), global_step)

            loss.backward()

            # compute loss and accuracy
            train_loss += loss.item() * inputs["convergent"].size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

            if use_tensorboard:
                log_param_stats(model, writer, global_step)

            if (step + 1) % n_steps_per_update == 0:
                if kwargs["parameters"]["use_clipping"]:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=kwargs["parameters"]["clip_max_norm"]
                    )
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            if use_tensorboard:
                # gradient norm (computed from current grads)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += float(param_norm.item()) ** 2
                writer.add_scalar("Batch/GradNorm", total_norm**0.5, global_step)

                # log LR (supports multiple param groups)
                for idx, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"LR/group{idx}", param_group["lr"], global_step)

        epoch_train_loss = train_loss / len(dataloaders["train"].dataset)
        epoch_train_acc = train_correct.double() / len(dataloaders["train"].dataset)

        # --- VALIDATION LOOP ---
        if use_val:
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in tqdm(
                    dataloaders["val"], desc=f"Epoch {epoch} [val]", leave=False
                ):
                    inputs = move_to_device(inputs, device)
                    labels = move_to_device(labels, device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if torch.isnan(loss):
                        logger.error(f"NaN loss detected at batch {step}")
                        continue

                    val_loss += loss.item() * inputs["convergent"].size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data)

            epoch_val_loss = val_loss / len(dataloaders["val"].dataset)
            epoch_val_acc = val_correct.double() / len(dataloaders["val"].dataset)

        epoch_time = time.time() - epoch_start

        # --- Epoch summary ---
        logger.info(f"\nEpoch {epoch}/{n_epochs} Summary:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        if use_val:
            logger.info(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        logger.info(f"  Time: {epoch_time:.2f}s")

        # --- Logging and checkpointing ---
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())
        if use_val:
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc.item())

        if use_tensorboard:
            writer.add_scalar("Epoch/Train_Loss", epoch_train_loss, epoch)
            writer.add_scalar("Epoch/Train_Acc", epoch_train_acc, epoch)

            if use_val:
                writer.add_scalar("Epoch/Val_Loss", epoch_val_loss, epoch)
                writer.add_scalar("Epoch/Val_Acc", epoch_val_acc, epoch)

            # --- Weight histograms (epoch-level) ---
            for name, param in model.named_parameters():
                writer.add_histogram(f"Weights/{name}", param, epoch)
                # epoch-level grads are often None; keep optional
                if param.grad is not None:
                    writer.add_histogram(f"Grads/{name}", param.grad, epoch)

        # Save checkpoint for every epoch
        torch.save(model.state_dict(), save_dir / f"{save_prefix}_epoch_{epoch}.pt")

        # Early stopping logic
        if use_val:
            improved = epoch_val_loss < best_val_loss
            if improved:
                epochs_no_improve = 0
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_weights = model.state_dict()
                torch.save(best_weights, save_dir / f"{save_prefix}_best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    logger.success(
                        f"⏹ Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs)"
                    )
                    model.load_state_dict(best_weights)
                    break
        else:
            # assume most recent epoch is best
            best_epoch = epoch

    if use_tensorboard:
        writer.close()

    return model, history, best_epoch


def compute_class_weights(train_ds):
    # Works with both Subset and raw dataset
    labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = class_sample_count.sum() / class_sample_count
    return torch.from_numpy(weight).float()


def get_splits(subjects, n_splits=5):

    # Create inner splits
    # Each inner split will be train*3, val, test
    inner_splits = []

    subj_splits = np.array_split(subjects, n_splits)
    duplicated_splits = subj_splits * 2

    for i in range(n_splits):

        val_subjs = duplicated_splits[i]
        test_subjs = duplicated_splits[i + 1]
        train_subjs = np.hstack(duplicated_splits[i + 2 : i + 5])

        inner_splits.append((train_subjs, val_subjs, test_subjs))

    return inner_splits


def log_param_stats(model, writer, step):
    """Logs per-parameter and global parameter norms and update norms."""
    total_param_norm = 0.0
    total_update_norm = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        # Parameter norm
        param_norm = p.data.norm(2)
        # Update norm (gradient norm scaled by LR later)
        update_norm = p.grad.data.norm(2)

        # Accumulate for overall stats
        total_param_norm += param_norm.item() ** 2
        total_update_norm += update_norm.item() ** 2

        # Per-parameter logging
        writer.add_scalar(f"ParamNorm/{name}", param_norm.item(), step)
        writer.add_scalar(f"UpdateNorm/{name}", update_norm.item(), step)
        writer.add_scalar(
            f"UpdateRatio/{name}",
            (update_norm / (param_norm + 1e-12)).item(),
            step,
        )

    # Global norms
    total_param_norm = total_param_norm**0.5
    total_update_norm = total_update_norm**0.5

    writer.add_scalar("Global/ParamNorm", total_param_norm, step)
    writer.add_scalar("Global/UpdateNorm", total_update_norm, step)
    writer.add_scalar("Global/UpdateRatio", total_update_norm / (total_param_norm + 1e-12), step)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_subject_indices(dataset, subj_list):
    """Get indices of samples in the dataset that belong to specified subjects.

    Args:
        dataset (_type_): _description_
        subj_list (list): list of subject identifiers to select

    Returns:
        list: list of indices corresponding to the specified subjects
    """
    return [i for i, s in enumerate(dataset.data) if s["subject"] in subj_list]


def build_model_optim_scheduler(model_type, hp, device, dataloaders=None, evaluate=False, **kwargs):

    if model_type == "Fusion":
        model = SEEGFusionModel(
            embed_dim=hp["embed_dim"],
            num_layers=hp["num_layers"],
            n_heads=hp["n_heads"],
            n_classes=2,
            device=device,
        )
        if not evaluate:
            if dataloaders is None:
                logger.error(f"dataloaders must be provided when not evaluating. Exiting ...")
                sys.exit(1)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=hp["base_lr"],
                weight_decay=hp["weight_decay"],
            )

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=hp["max_lr"],
                epochs=kwargs["parameters"]["n_epochs"],
                steps_per_epoch=len(dataloaders["train"]),
                pct_start=hp["pct_start"],
                anneal_strategy="cos",
            )
        else:
            optimizer = None
            scheduler = None
    else:
        logger.error(f"Only model_type == 'Fusion' is currently supported. Exiting ...")
        sys.exit(1)

    return model, optimizer, scheduler


def main(model_type, cohort_id, **kwargs):

    logger.info(f"Using cohort split {cohort_id}")

    # ---- Resolve subjects per split ----
    subjects = np.loadtxt("../../data/subjects.txt", dtype=str)
    shuffled_subjects = np.random.RandomState(SEED).permutation(subjects)
    splits = get_splits(shuffled_subjects, n_splits=kwargs["parameters"]["n_folds"])
    train_subjs, val_subjs, test_subjs = splits[cohort_id - 1]

    logger.info(f"Train subjects: {train_subjs}")
    logger.info(f"Val subjects: {val_subjs}")
    logger.info(f"Test subjects: {test_subjs}")

    # ---- Hyperparameter sweep ----
    best_val_loss = float("inf")
    best_state = None
    best_config = None
    results = []

    for hp_id in range(1, kwargs["parameters"]["n_hp_searches"] + 1):

        # Get hyperparameters
        hp = sample_hyperparameters(kwargs, cohort_id, hp_id)
        logger.info(f"HP set {hp_id}: {hp}")

        full_dataset = SEEGDataset(
            subjects=np.hstack((train_subjs, val_subjs)), embed_dim=hp["embed_dim"]
        )

        # ---- Build datasets ----
        train_ds = Subset(full_dataset, get_subject_indices(full_dataset, train_subjs))
        val_ds = Subset(full_dataset, get_subject_indices(full_dataset, val_subjs))

        dataloaders = {
            "train": DataLoader(
                train_ds,
                batch_size=kwargs["parameters"]["batch_size"],
                shuffle=True,
                num_workers=2,
                pin_memory=False,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=kwargs["parameters"]["batch_size"],
                shuffle=False,
                num_workers=2,
                pin_memory=False,
            ),
        }

        model, optimizer, scheduler = build_model_optim_scheduler(
            model_type, hp, device, dataloaders, **kwargs
        )

        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_ds).to(device))

        model, history, best_epoch = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_prefix=f"cohort{cohort_id}_hp{hp_id}",
            n_epochs=kwargs["parameters"]["n_epochs"],
            n_steps_per_update=kwargs["parameters"]["n_steps_per_update"],
            patience=kwargs["parameters"]["patience"],
            use_val=True,
            cohort_id=cohort_id,
            **kwargs,
        )

        val_loss = history["val_loss"][best_epoch - 1]

        results.append(
            {
                "cohort_id": cohort_id,
                "hp_id": hp_id,
                "val_loss": val_loss,
                "best_epoch": best_epoch,
                **hp,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_config = hp

        del model, train_ds, val_ds, full_dataset
        torch.cuda.empty_cache()

    # ---- Log hyperparameter results ----
    logger.success("Hyperparameter search complete!")
    logger.info("Results:")
    for result in results:
        logger.info(result)
    results_df = pd.DataFrame(results)
    outdir = Path(f"../../results/seed{SEED}")
    outdir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(
        outdir / f"{model_type}_seed{SEED}_cohort{cohort_id}_hp_search_results.csv", index=False
    )

    # ---- Test on best model ----
    logger.success(f"Best hyperparameters: {best_config}")
    save_dir = Path(f"../../experiments/seed{SEED}/cohort{cohort_id}")
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / f"cohort{cohort_id}_hp_best_overall.json", "w") as f:
        json.dump(best_config, f, indent=2)

    logger.info("Testing on best model from hyperparameter search.")

    model, _, _ = build_model_optim_scheduler(model_type, best_config, device, evaluate=True)

    model.load_state_dict(best_state)
    model.float()  # Convert to float32 for evaluation to avoid bfloat16/float32 type mismatch

    torch.save(model.state_dict(), save_dir / f"cohort{cohort_id}_best_overall_model.pt")

    all_metrics = []
    for subj in test_subjs:
        logger.info(f"Evaluating subject {subj}:")
        test_dataset = SEEGDataset(subjects=[subj], embed_dim=best_config["embed_dim"])
        metrics = evaluate_model(
            model,
            DataLoader(test_dataset, batch_size=kwargs["parameters"]["batch_size"], shuffle=False),
            device,
        )
        metrics.update(best_config)
        metrics["subj"] = subj
        del test_dataset
        all_metrics.append(metrics)

    first_column = ["subj"]
    all_columns = first_column + [k for k in all_metrics[0].keys() if k != "subj"]
    df = pd.DataFrame(all_metrics, columns=all_columns)
    df.to_csv(outdir / f"{model_type}_seed{SEED}_cohort{cohort_id}_test_results.csv", index=False)

    logger.success("Finished cohort run")


if __name__ == "__main__":

    parser = ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Fusion",
        help="Model type to train (e.g., 'Fusion' or 'Baseline'). Default is 'Fusion'.",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="../logs",
        help="Directory to save log files. Default is '../logs'.",
    )
    parser.add_argument(
        "-c",
        "--cohort",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Cohort split ID (1-5)",
    )

    args = parser.parse_args()
    model_type = args.model
    logdir = args.logdir
    cohort_id = args.cohort

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logdir) / f"train_{model_type}_{timestamp}.log"

    # Configure logger with process ID in format
    logger.add(
        log_file,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | PID:{process} | {level} | {message}",
        mode="w",
    )

    logger.info(f"Parameters in default.yaml are set as follows:")
    logger.info(open("../config/default.yaml").read())

    config = load_yaml()

    SEED = config["parameters"]["random_seed"]
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    main(model_type, cohort_id, **config)
