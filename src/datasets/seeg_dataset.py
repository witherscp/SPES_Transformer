import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from loguru import logger
import numpy as np
import math
import torch.nn as nn
from collections import OrderedDict


class SEEGDataset(Dataset):
    def __init__(
        self,
        subjects=None,
        data_dir="../../data/processed",
        transform=None,
        embed_dim=128,
        verbose=True,
        cache_size=50,  # Number of subjects to keep in memory
    ):
        """
        Memory-efficient SEEG dataset with LRU caching.

        Args:
            subjects (list[str], optional): List of subject IDs to include (without .pt extension).
            data_dir (str): Path to directory containing .pt files.
            transform (callable, optional): Optional transform to apply to each x.
            embed_dim (int): Embedding dimension for positional encoding.
            verbose (bool): Whether to log loading info.
            cache_size (int): Number of subjects to keep in memory (LRU cache).
        """
        self.transform = transform
        self.embed_dim = embed_dim
        self.pos_encoder = FourierPositionalEncoding(out_dim=embed_dim)
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.cache_size = cache_size

        # LRU cache for loaded subject data
        self._cache = OrderedDict()

        # If no subject list provided, include all .pt files
        if subjects is None:
            subjects = [f.stem for f in self.data_dir.glob("*.pt")]

        self.subjects = list(subjects)

        # Build metadata index: list of (subject, target_idx) tuples
        # Also collect max dimensions for padding
        self.sample_index = []
        self.max_stims = 0
        self.max_responses = 0
        self.max_trials = 0

        logger.info(f"Building metadata index for {len(self.subjects)} subjects...")
        for subj in self.subjects:
            path = self.data_dir / f"{subj}.pt"
            # Load just to get metadata, then discard
            subj_data = torch.load(path, weights_only=False)

            n_targets = len(subj_data["targets"])

            # Get dimensions from first target (assuming consistent across targets)
            x_conv = subj_data["convergent"]["data"][0]
            x_div = subj_data["divergent"]["data"][0]

            self.max_stims = max(self.max_stims, x_conv.shape[0])
            self.max_responses = max(self.max_responses, x_div.shape[0])
            self.max_trials = max(self.max_trials, x_conv.shape[1])

            if verbose:
                logger.info(
                    f"Indexed subject {subj}, "
                    f"n_stim={x_conv.shape[0]}, n_soz={subj_data['target_labels'].sum()}, "
                    f"n_resp={x_div.shape[0]}, n_targets={n_targets}"
                )

            for t in range(n_targets):
                self.sample_index.append((subj, t))

            del subj_data  # Free memory immediately

        logger.success(
            f"✅ Indexed {len(self.sample_index)} samples from {len(self.subjects)} subjects."
        )
        logger.info(
            f"Max dimensions: stims={self.max_stims}, responses={self.max_responses}, trials={self.max_trials}"
        )

    def _load_subject(self, subj):
        """Load and cache a subject's data."""
        if subj in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(subj)
            return self._cache[subj]

        # Load from disk
        path = self.data_dir / f"{subj}.pt"
        subj_data = torch.load(path, weights_only=False)

        # Process and store
        processed = self._process_subject(subj, subj_data)

        # Add to cache
        self._cache[subj] = processed

        # Evict oldest if cache is full
        while len(self._cache) > self.cache_size:
            evicted_subj = next(iter(self._cache))
            del self._cache[evicted_subj]
            if self.verbose:
                logger.info(f"Evicted subject {evicted_subj} from cache.")

        return processed

    def _process_subject(self, subj, subj_data):
        """Process a subject's data - store UNPADDED to save memory."""
        samples = {}

        conv_coords_raw = subj_data["convergent"]["stim_trial_coords"]
        div_coords_raw = subj_data["divergent"]["response_trial_coords"]

        # Encode coordinates once per subject
        conv_coord_mask = torch.all(torch.isnan(conv_coords_raw), dim=1)
        div_coord_mask = torch.all(torch.isnan(div_coords_raw), dim=1)
        conv_coords = self.pos_encoder(conv_coords_raw)
        div_coords = self.pos_encoder(div_coords_raw)
        conv_coords[conv_coord_mask] = torch.full((self.embed_dim,), np.nan)
        div_coords[div_coord_mask] = torch.full((self.embed_dim,), np.nan)

        # Store UNPADDED coordinates (save memory, pad lazily)
        conv_coords = torch.nan_to_num(conv_coords, nan=0).to(torch.float16)
        div_coords = torch.nan_to_num(div_coords, nan=0).to(torch.float16)

        samples["_shared"] = {
            "subject": subj,
            "convergent_coords": conv_coords,  # unpadded
            "divergent_coords": div_coords,    # unpadded
            "n_stims": conv_coords.shape[0],
            "n_responses": div_coords.shape[0],
        }

        n_targets = len(subj_data["targets"])

        for t in range(n_targets):
            x_conv = subj_data["convergent"]["data"][t]
            x_div = subj_data["divergent"]["data"][t]
            y = subj_data["target_labels"][t]

            # Store original shapes for lazy padding
            n_stims, n_trials_conv = x_conv.shape[:2]
            n_resp, n_trials_div = x_div.shape[:2]

            # Create masks for NaN values in original data only
            conv_nan_mask = torch.isnan(x_conv).all(dim=-1)
            div_nan_mask = torch.isnan(x_div).all(dim=-1)

            # Store UNPADDED data (much smaller for subjects with fewer channels)
            x_conv = torch.nan_to_num(x_conv, nan=0).to(torch.float16)
            x_div = torch.nan_to_num(x_div, nan=0).to(torch.float16)

            samples[t] = {
                "target_idx": t,
                "convergent": x_conv,       # unpadded
                "divergent": x_div,         # unpadded
                "conv_nan_mask": conv_nan_mask,
                "div_nan_mask": div_nan_mask,
                "n_trials": n_trials_conv,
                "label": y,
            }

        return samples

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        subj, target_idx = self.sample_index[idx]

        subj_samples = self._load_subject(subj)
        sample = subj_samples[target_idx]
        shared = subj_samples["_shared"]

        # Lazy padding at access time
        x_conv = sample["convergent"]
        x_div = sample["divergent"]
        conv_coords = shared["convergent_coords"]
        div_coords = shared["divergent_coords"]

        # Pad data tensors
        x_conv_padded = torch.nn.functional.pad(
            x_conv,
            (0, 0, 0, self.max_trials - x_conv.shape[1], 0, self.max_stims - x_conv.shape[0]),
            value=0,
        )
        x_div_padded = torch.nn.functional.pad(
            x_div,
            (0, 0, 0, self.max_trials - x_div.shape[1], 0, self.max_responses - x_div.shape[0]),
            value=0,
        )

        # Pad coordinates
        conv_coords_padded = torch.nn.functional.pad(
            conv_coords, (0, 0, 0, self.max_stims - conv_coords.shape[0]), value=0
        )
        div_coords_padded = torch.nn.functional.pad(
            div_coords, (0, 0, 0, self.max_responses - div_coords.shape[0]), value=0
        )

        # Build masks (padded regions + original NaN positions)
        conv_mask = torch.ones(self.max_stims, self.max_trials, dtype=torch.bool)
        div_mask = torch.ones(self.max_responses, self.max_trials, dtype=torch.bool)
        
        n_s, n_t = x_conv.shape[:2]
        n_r = x_div.shape[0]
        
        conv_mask[:n_s, :n_t] = sample["conv_nan_mask"]
        div_mask[:n_r, :n_t] = sample["div_nan_mask"]

        x = {
            "convergent": x_conv_padded.float(),
            "divergent": x_div_padded.float(),
            "convergent_mask": conv_mask,
            "divergent_mask": div_mask,
            "convergent_coords": conv_coords_padded.float(),
            "divergent_coords": div_coords_padded.float(),
        }

        return x, sample["label"]

    # For compatibility with get_subject_indices and compute_class_weights
    @property
    def data(self):
        """Returns a lightweight list for indexing by subject (lazy)."""
        return [{"subject": subj, "label": None} for subj, _ in self.sample_index]


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