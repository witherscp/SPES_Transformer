from datetime import datetime
import time
from tqdm import tqdm
import random
from pathlib import Path
from loguru import logger
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from src.utils import load_yaml, move_to_device
from src.datasets.seeg_dataset import SEEGDataset
from src.models.model import SEEGFusionModel, BaselineModel
from src.training.evaluate import evaluate_model

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    save_prefix,
    scheduler=None,
    n_epochs=50,
    n_steps_per_update=1,
    patience=2,
    use_val=True,
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
        scheduler: learning rate scheduler (optional)
        n_epochs: int, number of epochs (default: 50)
        n_steps_per_update: gradient accumulation steps (default: 1)
        patience: early stopping patience (# consecutive epochs without improvement) (default: 2)
        use_val: whether to use validation set for early stopping (default: True)

    Returns:
        model: trained model (with best weights loaded)
        history: dict with losses and accuracies
        best_epoch: epoch with best validation loss
    """
    save_dir = Path(f"../experiments")
    save_dir.mkdir(exist_ok=True, parents=True)
    model.to(device)

    # --- TensorBoard setup ---
    tb_run_name = f"{save_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"../tb_logs/{tb_run_name}")

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
            writer.add_scalar("Batch/Loss", loss.item(), global_step)

            loss.backward()

            # compute loss and accuracy
            train_loss += loss.item() * inputs["convergent"].size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

            log_param_stats(model, writer, step)

            if (step + 1) % n_steps_per_update == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

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
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_weights = model.state_dict()
                torch.save(best_weights, save_dir / f"{save_prefix}_best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.success(
                        f"⏹ Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs)"
                    )
                    model.load_state_dict(best_weights)
                    break
        else:
            # assume most recent epoch is best
            best_epoch = epoch

    writer.close()

    return model, history, best_epoch


def compute_class_weights(train_ds):
    labels = np.array([v[1] for v in train_ds])
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = class_sample_count.sum() / class_sample_count
    return torch.from_numpy(weight).float()


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
    total_param_norm = total_param_norm ** 0.5
    total_update_norm = total_update_norm ** 0.5

    writer.add_scalar("Global/ParamNorm", total_param_norm, step)
    writer.add_scalar("Global/UpdateNorm", total_update_norm, step)
    writer.add_scalar("Global/UpdateRatio",
                      total_update_norm / (total_param_norm + 1e-12),
                      step)


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


def main(model_type, **kwargs):

    # Load dataset
    sz_free_subjects = [
        "Epat31",
        "Epat35",
        "Epat37",
        "Epat38",
        "Spat31",
        "Spat37",
        "Spat41",
        "Spat42",
    ]
    if kwargs["Parameters"]["sz_free_only"]:
        full_dataset = SEEGDataset(subjects=sz_free_subjects)
    else:
        full_dataset = SEEGDataset()

    n_inner_splits = 5
    val_ratio = kwargs["Parameters"]["val_ratio"]
    if val_ratio > 0:
        use_val = True
    else:
        use_val = False

    metric_dict = {}
    for test_subj in full_dataset.subjects:
        logger.info(f"\n=== Test subject: {test_subj} ===")
        remaining_subjs = [s for s in full_dataset.subjects if s != test_subj]

        # Outer split: test vs remaining
        test_idx = get_subject_indices(full_dataset, [test_subj])
        test_ds = Subset(full_dataset, test_idx)

        # Shuffle remaining subjects so different folds vary
        random.shuffle(remaining_subjs)

        if use_val:
            # Create inner splits
            inner_splits = []
            # Split based on val_ratio
            n_val_subjs = int(len(remaining_subjs) * val_ratio)
            for i in range(n_inner_splits):
                # rotate subjects for different validation sets
                if (i + 1) * n_val_subjs > len(remaining_subjs):
                    val_subjs = remaining_subjs[i * n_val_subjs :]
                else:
                    val_subjs = remaining_subjs[i * n_val_subjs : (i + 1) * n_val_subjs]
                train_subjs = [s for s in remaining_subjs if s not in val_subjs]
                inner_splits.append((train_subjs, val_subjs))
        else:
            inner_splits = [(remaining_subjs, [])]

        # Run inner CV for this test subject
        for k, (train_subjs, val_subjs) in enumerate(inner_splits):
            logger.info(f"\nInner split {k+1}: train={train_subjs}, val={val_subjs}")

            train_idx = get_subject_indices(full_dataset, train_subjs)
            val_idx = get_subject_indices(full_dataset, val_subjs)

            train_ds = Subset(full_dataset, train_idx)
            val_ds = Subset(full_dataset, val_idx)

            dataloaders = {
                "train": DataLoader(
                    train_ds,
                    batch_size=kwargs["Parameters"]["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
                "val": DataLoader(
                    val_ds,
                    batch_size=kwargs["Parameters"]["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
                "test": DataLoader(
                    test_ds,
                    batch_size=kwargs["Parameters"]["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
            }

            weights = compute_class_weights(train_ds)

            if model_type == "Fusion":
                model = SEEGFusionModel(
                    embed_dim=kwargs["Parameters"]["embed_dim"], n_classes=2, device=device
                )
            elif model_type == "Baseline":
                model = BaselineModel(
                    embed_dim=kwargs["Parameters"]["embed_dim"],
                    n_classes=2,
                    device=device,
                    stim_model="convergent",
                    n_elecs=25,
                    generator=g,
                )
            model.to(device)

            optimizer = optim.AdamW(
                model.parameters(), 
                lr=kwargs["Parameters"]["base_lr"],
                weight_decay=kwargs['Parameters']['weight_decay']
            )
            # scheduler = optim.lr_scheduler.CyclicLR(
            #     optimizer,
            #     base_lr=kwargs["Parameters"]["base_lr"],
            #     max_lr=kwargs["Parameters"]["max_lr"],
            #     step_size_up=kwargs["Parameters"]["step_size_up_down"],
            #     step_size_down=kwargs["Parameters"]["step_size_up_down"],
            #     cycle_momentum=False,
            #     mode='triangular'
            # )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=kwargs['Parameters']['max_lr'],
                epochs=kwargs['Parameters']['n_epochs'],
                steps_per_epoch=len(dataloaders['train']),
                pct_start=kwargs['Parameters']['pct_start'],
                anneal_strategy='cos'
            )

            criterion = nn.CrossEntropyLoss(weight=weights.to(device))

            model, _, best_epoch = train_model(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                save_prefix=f"{test_subj}_model_{model_type}_split_{k+1}",
                n_epochs=kwargs["Parameters"]["n_epochs"],
                patience=kwargs["Parameters"]["patience"],
                n_steps_per_update=kwargs["Parameters"]["n_steps_per_update"],
                use_val=use_val
            )

            logger.success(
                f"✅ Training completed for test subject {test_subj}, inner split {k+1}. Best epoch: {best_epoch}"
            )

            metrics = evaluate_model(model, dataloaders["test"], device)
            metric_dict[f"{test_subj}_split_{k+1}"] = metrics

    logger.success(f"\n=== Summary of metrics across all test subjects and splits ===")
    for key, value in metric_dict.items():
        logger.success(f"{key}: {value}")


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

    args = parser.parse_args()
    model_type = args.model
    logdir = args.logdir

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

    main(model_type, **config)
