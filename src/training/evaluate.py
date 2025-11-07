import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.training.train import move_to_device
from loguru import logger


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate a model on test data and compute AUROC.
    Args:
        model: trained nn.Module
        dataloader: DataLoader for test set
        criterion: loss function
        device: torch device
    Returns:
        test_loss, test_acc, test_auc
    """
    model.eval()
    model.to(device)

    test_loss = 0.0
    correct = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs["convergent"].size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # assuming binary classification
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = test_loss / len(dataloader.dataset)
    test_acc = correct.double() / len(dataloader.dataset)

    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        test_auc = float("nan")  # handle single-class case

    logger.info(f"\nTest Summary:")
    logger.info(f"  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUROC: {test_auc:.4f}")

    return test_loss, test_acc.item(), test_auc
