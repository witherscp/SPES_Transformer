import torch
from src.utils import move_to_device
from loguru import logger
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a given dataset and compute AUROC, F1 score, and Youden index.

    Returns:
        dict with accuracy, AUROC, F1, sensitivity, specificity, Youden index
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = move_to_device(inputs, device)
            labels = labels.to(device)

            outputs = model(inputs)

            # For probabilities and predictions
            probs = torch.softmax(outputs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(
                probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy()
            )

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = np.nan  # e.g., if only one class present in labels

    # Compute ROC curve and find best threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    youden_values = tpr - fpr
    best_idx = np.argmax(youden_values)
    best_threshold = thresholds[best_idx]
    youden_index = youden_values[best_idx]
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]

    # Compute accuracy and F1 at that threshold
    preds = (all_probs >= best_threshold).astype(int)
    acc = np.mean(preds == all_labels)
    f1 = f1_score(all_labels, preds)

    metrics = {
        "accuracy": acc,
        "auroc": auroc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_index": youden_index,
        "optimal_threshold": best_threshold,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    logger.success(metrics)

    return metrics
