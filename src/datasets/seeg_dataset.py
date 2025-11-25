import torch
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger
import numpy as np
import math
import torch.nn as nn


class SEEGDataset(Dataset):
    def __init__(
        self, subjects=None, data_dir="../../data/processed", transform=None, embed_dim=128
    ):
        """
        Args:
            subjects (list[str], optional): List of subject IDs to include (without .pt extension).
                                             If None, all .pt files in directory are loaded.
            data_dir (str): Path to directory containing .pt files.
            transform (callable, optional): Optional transform to apply to each x.
        """

        self.data = []
        self.transform = transform
        self.embed_dim = embed_dim
        self.pos_encoder = FourierPositionalEncoding(out_dim=embed_dim)
        data_dir = Path(data_dir)

        # If no subject list provided, include all .pt files
        if subjects is None:
            subjects = [f.stem for f in data_dir.glob("*.pt")]

        self.subjects = subjects

        max_stims, max_responses, max_trials = 0, 0, 0
        for subj in subjects:
            path = data_dir / f"{subj}.pt"
            subj_data = torch.load(path, weights_only=False)

            logger.info(
                f"Loading subject {subj}, n_targets={len(subj_data['targets'])}, "
                f"n_soz = {subj_data['target_labels'].sum()}, n_stim={subj_data['convergent']['data'][0].shape[0]}, "
                f"n_resp={subj_data['divergent']['data'][0].shape[0]}, "
            )

            # Each subject may have multiple electrodes (targets)
            n_targets = len(subj_data["targets"])
            conv_currents = subj_data["convergent"]["stim_trial_currents"]  # [n_stims, n_trials]
            conv_coords = subj_data["convergent"]["stim_trial_coords"]  # [n_stims, 3]
            div_coords = subj_data["divergent"]["response_trial_coords"]  # [n_responses, 3]

            for t in range(n_targets):
                x_conv = subj_data["convergent"]["data"][t]  # e.g., [n_stims, n_trials, n_times]
                x_div = subj_data["divergent"]["data"][t]  # e.g., [n_responses, n_trials, n_times]
                y = subj_data["target_labels"][t]

                # reshape div_currents to [n_responses, n_trials]
                div_currents = subj_data["divergent"]["target_trial_currents"][t]  # [n_trials]
                div_currents = torch.tile(
                    div_currents, dims=(x_div.shape[0], 1)
                )  # [n_responses, n_trials]

                max_stims = max(max_stims, x_conv.shape[0])
                max_responses = max(max_responses, x_div.shape[0])
                max_trials = max(
                    max_trials, x_conv.shape[1]
                )  # assuming conv and div have same n_trials

                # coordinate masks
                conv_coord_mask = torch.all(torch.isnan(conv_currents), dim=1)
                div_coord_mask = torch.all(torch.isnan(div_currents), dim=1)
                conv_coords = self.pos_encoder(conv_coords)
                div_coords = self.pos_encoder(div_coords)
                conv_coords[conv_coord_mask] = torch.full((self.embed_dim,), np.nan)
                div_coords[div_coord_mask] = torch.full((self.embed_dim,), np.nan)

                sample = {
                    "subject": subj,
                    "target_idx": t,
                    "convergent": x_conv,
                    "divergent": x_div,
                    "convergent_currents": conv_currents,
                    "divergent_currents": div_currents,
                    "convergent_coords": conv_coords,
                    "divergent_coords": div_coords,
                    "label": y,
                }

                self.data.append(sample)

        for sample in self.data:
            # Now apply padding to all samples to have uniform sizes
            # pads last dimension before and after by 0,0
            # pads 2nd dimension (n_trials) by 0 before and (max_trials - current) after
            # pads 1st dimension (n_stims or n_responses) by 0 before and (max_stims - current) or (max_responses - current) after
            sample["divergent"] = torch.nn.functional.pad(
                sample["divergent"],
                (
                    0,
                    0,
                    0,
                    max_trials - sample["divergent"].shape[1],
                    0,
                    max_responses - sample["divergent"].shape[0],
                ),
                value=np.nan,
            )
            sample["convergent"] = torch.nn.functional.pad(
                sample["convergent"],
                (
                    0,
                    0,
                    0,
                    max_trials - sample["convergent"].shape[1],
                    0,
                    max_stims - sample["convergent"].shape[0],
                ),
                value=np.nan,
            )
            sample["divergent_currents"] = torch.nn.functional.pad(
                sample["divergent_currents"],
                (
                    0,
                    max_trials - sample["divergent_currents"].shape[1],
                    0,
                    max_responses - sample["divergent_currents"].shape[0],
                ),
                value=0,
            )
            sample["convergent_currents"] = torch.nn.functional.pad(
                sample["convergent_currents"],
                (
                    0,
                    max_trials - sample["convergent_currents"].shape[1],
                    0,
                    max_stims - sample["convergent_currents"].shape[0],
                ),
                value=0,
            )
            sample["divergent_coords"] = torch.nn.functional.pad(
                sample["divergent_coords"],
                (
                    0,
                    0,
                    0,
                    max_responses - sample["divergent_coords"].shape[0],
                ),
                value=0,
            )
            sample["convergent_coords"] = torch.nn.functional.pad(
                sample["convergent_coords"],
                (
                    0,
                    0,
                    0,
                    max_stims - sample["convergent_coords"].shape[0],
                ),
                value=0,
            )

            # Create padding masks
            sample["convergent_mask"] = torch.isnan(sample["convergent"]).all(
                dim=-1
            )  # [n_stims, n_trials]
            sample["divergent_mask"] = torch.isnan(sample["divergent"]).all(
                dim=-1
            )  # [n_responses, n_trials]

            # Replace nan with zeros for model input
            sample["convergent"] = torch.nan_to_num(sample["convergent"], nan=0)
            sample["divergent"] = torch.nan_to_num(sample["divergent"], nan=0)
            sample["convergent_currents"] = torch.nan_to_num(sample["convergent_currents"], nan=0)
            sample["divergent_currents"] = torch.nan_to_num(sample["divergent_currents"], nan=0)
            sample["convergent_coords"] = torch.nan_to_num(sample["convergent_coords"], nan=0)
            sample["divergent_coords"] = torch.nan_to_num(sample["divergent_coords"], nan=0)

        logger.success(f"✅ Loaded {len(self.data)} total samples from {len(subjects)} subjects.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        x = {
            "convergent": sample["convergent"],
            "divergent": sample["divergent"],
            "convergent_mask": sample["convergent_mask"],
            "divergent_mask": sample["divergent_mask"],
            "convergent_coords": sample["convergent_coords"],
            "divergent_coords": sample["divergent_coords"],
        }
        y = sample["label"]

        # if self.transform:
        #     x = self.transform(x)

        return x, y


class FourierPositionalEncoding(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.num_freqs = out_dim // (3 * 2)  # since we have sin and cos for each of x,y,z
        self.out_dim = out_dim

        # frequencies: [1, 2, 4, 8, ...] * π
        self.register_buffer("freq_bands", (2.0 ** torch.arange(self.num_freqs)) * math.pi)

    def forward(self, coords):
        """
        coords: (..., 3) tensor of (x,y,z) coordinates
        returns: (..., out_dim) tensor
        """
        # coords: (..., 3) → (..., 1, 3)
        c = coords.unsqueeze(-2)  # add freq dim

        # apply frequencies: (..., num_freqs, 3)
        fc = c * self.freq_bands.view(1, -1, 1)

        # sin/cos features: (..., num_freqs*2*3)
        pe = torch.cat([fc.sin(), fc.cos()], dim=-1).reshape(*coords.shape[:-1], -1)

        # pad or truncate to target out_dim
        if pe.shape[-1] < self.out_dim:
            pad = torch.zeros(*pe.shape[:-1], self.out_dim - pe.shape[-1], device=pe.device)
            pe = torch.cat([pe, pad], dim=-1)
        else:
            pe = pe[..., : self.out_dim]

        return pe
