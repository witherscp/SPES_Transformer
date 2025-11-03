import torch
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger


class SEEGDataset(Dataset):
    def __init__(self, subjects=None, data_dir="../data/processed", transform=None):
        """
        Args:
            subjects (list[str], optional): List of subject IDs to include (without .pt extension).
                                             If None, all .pt files in directory are loaded.
            data_dir (str): Path to directory containing .pt files.
            transform (callable, optional): Optional transform to apply to each x.
        """

        self.data = []
        self.transform = transform
        data_dir = Path(data_dir)

        # If no subject list provided, include all .pt files
        if subjects is None:
            subjects = [f.stem for f in data_dir.glob("*.pt")]

        for subj in subjects:
            path = data_dir / f"{subj}.pt"
            subj_data = torch.load(path, weights_only=False)

            # Each subject may have multiple electrodes (targets)
            n_targets = len(subj_data["targets"])

            for t in range(n_targets):
                x_conv = subj_data["convergent"]["data"][t]  # e.g., [n_stims, n_trials, n_times]
                x_div = subj_data["divergent"]["data"][t]  # e.g., [n_responses, n_trials, n_times]
                y = subj_data["target_labels"][t]

                sample = {
                    "subject": subj,
                    "target_idx": t,
                    "convergent": x_conv,
                    "divergent": x_div,
                    "label": y,
                }

                self.data.append(sample)

        logger.success(f"✅ Loaded {len(self.data)} total samples from {len(subjects)} subjects.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        x = {"convergent": sample["convergent"], "divergent": sample["divergent"]}
        y = sample["label"]

        # if self.transform:
        #     x = self.transform(x)

        return x, y
