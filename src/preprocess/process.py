import torch
from loguru import logger
from pathlib import Path


@logger.catch
def gather_data(subj):
    """
    Dummy implementation of gather_data function.
    In actual implementation, this function would process .mat files
    and create a .pt file for the given subject.
    """

    # Simulate data processing and saving
    data = {"subject": subj, "data": [1, 2, 3]}  # Dummy data

    data_dir = Path("../../data/raw")
    torch.save(data, data_dir / f"{subj}_data.pt")
    return True
