import torch
import time
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    save_prefix,
    n_epochs=50,
    patience=2,
):
    """
    Train a model with early stopping and per-epoch checkpoint saving.
    Args:
        model: nn.Module
        dataloaders: dict with 'train' and 'val' DataLoaders
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        device: torch device
        save_prefix: prefix for saving out model weights
        n_epochs: int, number of epochs
        patience: early stopping patience (# consecutive epochs without improvement)

    Returns:
        model: trained model (with best weights loaded)
        history: dict with losses and accuracies
        best_epoch: epoch with best validation loss
    """
    save_dir = Path(f"../experiments")
    save_dir.mkdir(exist_ok=True, parents=True)
    model.to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"\nStarting training for {n_epochs} epochs on device: {device}")
    print("=" * 60)

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        train_correct = 0

        # --- TRAIN LOOP ---
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(dataloaders["train"], desc=f"Epoch {epoch} [train]", leave=False)
        ):
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                logger.error(f"NaN loss detected at batch {batch_idx}")
                continue

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs["convergent"].size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        epoch_train_loss = train_loss / len(dataloaders["train"].dataset)
        epoch_train_acc = train_correct.double() / len(dataloaders["train"].dataset)

        # --- VALIDATION LOOP ---
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

        logger.info(f"\nEpoch {epoch}/{n_epochs} Summary:")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        logger.info(f"  Time: {epoch_time:.2f}s")
        print("=" * 60)

        # --- Logging and checkpointing ---
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())

        # Save checkpoint for every epoch
        torch.save(model.state_dict(), save_dir / f"{save_prefix}_epoch_{epoch}.pt")

        # Early stopping logic
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
                break

    # Load best weights before returning
    model.load_state_dict(best_weights)
    return model, history, best_epoch


def move_to_device(obj, device):
    """
    Recursively moves tensors in a dictionary, list, or tuple to a specified device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list) or isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return type(obj)(res)
    else:
        # Return non-tensor objects as they are (e.g., int, str, etc.)
        return obj
