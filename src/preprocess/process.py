from collections import Counter

import torch
from loguru import logger
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils import (
    get_subj_path,
    load_yaml,
    get_soz_label_dict,
    get_mni_coords_dict,
    loadmat,
    calc_euc_distance,
    load_spes_data,
)


@logger.catch
def build_subject_pt(subj, **kwargs):
    """Build .pt file for a given subject from their .mat train files.

    Args:
        subj (str): Subject identifier.

    Returns:
        bool: True if the .pt file was successfully created, False otherwise.
    """

    ## ------ Gather relevant metadata ------

    # get soz label dict
    soz_label_dict = get_soz_label_dict(subj)

    # get mni coords dict
    mni_coords_dict = get_mni_coords_dict(subj)

    # check existence of both dictionaries
    if (not soz_label_dict) or (not mni_coords_dict):
        logger.error("Patient missing SOZ labels or MNI coords. Cannot proceed.")
        return False

    ## ------ Load and format pulse data ------

    # Get paths to all .mat train files
    spes_data_dict = load_spes_data(subj)

    # check existence of train paths
    if len(spes_data_dict) == 0:
        return False

    # check that labels are consistent with soz_label_dict
    not_present = []
    for label in soz_label_dict.keys():
        if label in spes_data_dict["labels"]:
            continue
        else:
            not_present.append(label)
    if not_present:
        logger.error(
            f"There are {len(not_present)} channels present in SOZ label dict that "
            f"are not present in spes_data_dict['labels']. These include: {sorted(not_present)}."
            " This should not happen, so something must have gone wrong."
        )
        return False

    # Load convergent and divergent datasets
    data_dict = combine_pulses(
        spes_data_dict=spes_data_dict, mni_coords_dict=mni_coords_dict, **kwargs
    )

    ## ------ Save processed data as .pt file ------
    # get target_labels
    target_labels = [
        1 if soz_label_dict.get(target, "NIZ") == "SOZ" else 0 for target in data_dict["targets"]
    ]

    # Simulate data processing and saving
    data = {
        "subject": subj,
        "targets": data_dict["targets"],
        "convergent": {
            "data": torch.tensor(data_dict["convergent"]["data"], dtype=torch.float32),
            "stim_names": data_dict["convergent"]["stim_names"],
            "stim_trial_currents": torch.tensor(
                data_dict["convergent"]["stim_trial_currents"], dtype=torch.float32
            ),
            "stim_trial_coords": torch.tensor(
                data_dict["convergent"]["stim_trial_coords"], dtype=torch.float32
            ),
        },
        "divergent": {
            "data": torch.tensor(data_dict["divergent"]["data"], dtype=torch.float32),
            "response_names": data_dict["divergent"]["response_names"],
            "target_trial_currents": torch.tensor(
                data_dict["divergent"]["target_trial_currents"], dtype=torch.float32
            ),
            "response_trial_coords": torch.tensor(
                data_dict["divergent"]["response_trial_coords"], dtype=torch.float32
            ),
        },
        "target_labels": torch.tensor(target_labels, dtype=torch.long),
    }

    data_dir = Path("../../data/processed")
    torch.save(data, data_dir / f"{subj}.pt")
    return True


def combine_pulses(spes_data_dict, mni_coords_dict, **kwargs):
    """
    Combine convergent and divergent pulse data from multiple .mat files.
    Args:
        spes_data_dict (dict): SPES data loaded from .mat files
        possible_targets (list of str): List of possible target electrode names.

    Returns:
        dict: A dictionary with keys 'convergent', 'divergent', and 'targets'.
            'convergent' contains keys:
                - data: np.array [n_targets, n_stims, n_trials, n_timepoints]
                - stim_names: np.array [n_stims]
                - stim_trial_currents: np.array [n_stims, n_trials]
            'divergent' contains keys:
                - data: np.array [n_targets, n_responses, n_trials, n_timepoints]
                - response_names: np.array [n_responses]
                - target_trial_currents: np.array [n_targets, n_trials]
            targets: np.array [n_targets]
    """

    # get stimulated electrodes from all files
    fs = spes_data_dict["fs"]
    stim_data = spes_data_dict["stim_data"]

    # get number of trials
    trial_counter = {}
    for stim_trial_block, data_dict in stim_data.items():
        stim = stim_trial_block.split("_")[0]
        trial_counter.setdefault(stim, 0)
        trial_counter[stim] += data_dict["pulse_times"].size
    n_trials = 0
    for v in trial_counter.values():
        if v > n_trials:
            n_trials = v

    unique_stims = np.array(list(trial_counter.keys()))
    targets = unique_stims  # any stim electrode is a target for classification

    # calculate number of timepoints
    win_start = int(kwargs["parameters"]["artifact_pad_ms"] / 1000 * fs)
    win_end = int((1000 - kwargs["parameters"]["artifact_pad_ms"]) / 1000 * fs)
    n_timepoints = win_end - win_start

    # get response names
    response_names = spes_data_dict["labels"]

    # get remaining dimensions
    n_targets = len(targets)
    n_stims = len(unique_stims)  # number of unique stim electrodes
    n_responses = len(response_names)  # number of possible response electrodes

    convergent = {
        "data": np.full(
            (n_targets, n_stims, n_trials, n_timepoints), np.nan
        ),  # [n_targets, n_stims, n_trials, n_timepoints]
        "stim_names": unique_stims,  # [n_stims]
        "stim_trial_currents": np.full((n_stims, n_trials), np.nan),  # [n_stims, n_trials],
        "stim_trial_coords": np.array([mni_coords_dict[s] for s in unique_stims]),  # [n_stims, 3]
    }

    divergent = {
        "data": np.full(
            (n_targets, n_responses, n_trials, n_timepoints), np.nan
        ),  # [n_targets, n_responses, n_trials, n_timepoints]
        "response_names": response_names,  # [n_responses]
        "target_trial_currents": np.full((n_targets, n_trials), np.nan),  # [n_targets, n_trials]
        "response_trial_coords": np.array(
            [mni_coords_dict[r] for r in response_names]
        ),  # [n_responses, 3]
    }

    stim_trial_idx_dict = {}

    for stim_block_train, data_dict in stim_data.items():
        logger.info(f"Working on {stim_block_train}")
        stim = stim_block_train.split("_")[0]
        stim_i = np.where(unique_stims == stim)[0][0]
        dists_stim_to_targets = np.array(
            [calc_euc_distance(mni_coords_dict, stim, t) for t in targets]
        )
        stim_trial_idx_dict.setdefault(stim, -1)

        # get baseline mean and standard deviation
        baseline_end_idx = int(kwargs["parameters"]["zscore_baseline_end_s"] * fs)
        baseline_period = data_dict["data"][:, :baseline_end_idx]
        baseline_mean = np.mean(baseline_period, axis=1, keepdims=True)
        baseline_std = np.std(baseline_period, axis=1, keepdims=True)

        # get convergent mask
        convergent_mask = (
            dists_stim_to_targets >= kwargs["parameters"]["dist_threshold"]
        )  # only consider targets greater than dist_threshold from stim
        label_indices = np.array(
            [np.where(response_names == t)[0][0] for t in targets[convergent_mask]]
        )

        # get divergent mask
        dists_stim_to_responses = np.array(
            [calc_euc_distance(mni_coords_dict, stim, r) for r in response_names]
        )
        divergent_mask = (
            dists_stim_to_responses >= kwargs["parameters"]["dist_threshold"]
        )  # only consider responses greater than 20mm from stim
        target_idx = np.where(targets == stim)[0][0]

        for pulse_time in data_dict["pulse_times"]:
            stim_trial_idx_dict[stim] += 1
            trial_idx = stim_trial_idx_dict[stim]

            # select pulse window
            pulse_time_idx = int(pulse_time * fs)
            pulse_window = data_dict["data"][
                :, pulse_time_idx + win_start : pulse_time_idx + win_end
            ]

            # normalize pulse
            norm_pulse_window = (pulse_window - baseline_mean) / baseline_std

            # get current
            current = data_dict["current"]

            # Extract convergent data
            convergent["data"][convergent_mask, stim_i, trial_idx, :] = norm_pulse_window[
                label_indices, :
            ]
            convergent["stim_trial_currents"][stim_i, trial_idx] = current

            # Extract divergent data
            divergent["data"][target_idx, divergent_mask, trial_idx, :] = norm_pulse_window[
                divergent_mask, :
            ]
            divergent["target_trial_currents"][target_idx, trial_idx] = current

    # Check if a target has no convergent responses or no divergent responses
    if np.any(np.all(np.isnan(convergent["data"][:, :, 0, 0]), axis=1)):
        logger.error(
            "Targets are present that have no convergent responses (likely due to distance threshold). Need to implement removal!"
        )
    elif np.any(np.all(np.isnan(divergent["data"][:, :, 0, 0]), axis=1)):
        logger.error(
            "Targets are present that have no divergent responses (likely due to distance threshold). Need to implement removal!"
        )

    out_dict = {"convergent": convergent, "divergent": divergent, "targets": targets}

    return out_dict
