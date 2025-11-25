import torch
from loguru import logger
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils import (
    get_subj_path,
    load_yaml,
    get_dks_region_dict,
    get_dks_broad_region_dict,
    get_soz_label_dict,
    get_mni_coords_dict,
    loadmat,
    calc_euc_distance,
)


@logger.catch
def build_subject_pt(subj, **kwargs):
    """
    Dummy implementation of build_subject_pt function.
    In actual implementation, this function would process .mat files
    and create a .pt file for the given subject.
    """

    ## ------ Gather relevant metadata ------

    # get soz label dict
    soz_label_dict = get_soz_label_dict(subj)

    # # get dks region dict
    # dks_region_dict = get_dks_region_dict(subj)

    # # get dks broad region dict
    # dks_broad_region_dict = get_dks_broad_region_dict(subj)

    # get mni coords dict
    mni_coords_dict = get_mni_coords_dict(subj)

    ## ------ Load and format pulse data ------

    # Get paths to all .mat pulse files
    pulse_paths = get_pulse_paths(subj)

    # Load convergent and divergent datasets
    data_dict = combine_pulses(
        pulse_paths=pulse_paths,
        possible_targets=soz_label_dict.keys(),  # most restrictive set of electrodes (must know for electrode to be target)
        mni_coords_dict=mni_coords_dict,
        dist_threshold=kwargs["Parameters"][
            "dist_threshold"
        ],  # only consider electrodes greater than 20mm from stim
    )

    ## ------ Save processed data as .pt file ------

    # get target_labels
    target_labels = [1 if soz_label_dict[target] == "SOZ" else 0 for target in data_dict["targets"]]

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


def combine_pulses(pulse_paths, possible_targets, mni_coords_dict, dist_threshold=20):
    """
    Combine convergent and divergent pulse data from multiple .mat files.
    Args:
        pulse_paths (list of str): List of paths to .mat pulse files.
        possible_targets (list of str): List of possible target electrode names.
        dist_threshold (float): Minimum distance (in mm) between stim and target electrodes
            for inclusion. Default is 20mm.

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

    # get subject name for later use
    subj = pulse_paths[0].name.split("_")[0]

    # get stimulated electrodes from all files
    stims_lst = [path.name.split("_")[1] for path in pulse_paths]
    unique_stims = list(set(stims_lst))
    targets = np.array(list(set(unique_stims) & set(possible_targets)))

    # get maximum number of pulses across all stim electrodes
    n_trials = 0  # number of possible trials per stim electrode
    for stim in unique_stims:
        stim_paths = [path for path in pulse_paths if f"_{stim}_" in path.name]
        n_trials = max(n_trials, len(list(stim_paths)))

    # get number of timepoints and response_names based on first file
    # (assuming all files have same number of timepoints and labels)
    mat = loadmat(pulse_paths[0], data=["labels", "data"])
    n_timepoints = mat["data"].shape[1]
    response_names = mat["labels"]

    # get remaining dimensions
    n_targets = len(
        targets
    )  # the intersection of response_elecs and stimulated electrodes across files
    n_stims = len(unique_stims)  # number of unique stim electrodes across files
    n_responses = len(response_names)  # number of possible response electrodes

    convergent = {
        "data": np.full(
            (n_targets, n_stims, n_trials, n_timepoints), np.nan
        ),  # [n_targets, n_stims, n_trials, n_timepoints]
        "stim_names": np.array(unique_stims),  # [n_stims]
        "stim_trial_currents": np.full((n_stims, n_trials), np.nan),  # [n_stims, n_trials],
        "stim_trial_coords": np.array([mni_coords_dict[s] for s in unique_stims]),  # [n_stims, 3]
    }

    divergent = {
        "data": np.full(
            (n_targets, n_responses, n_trials, n_timepoints), np.nan
        ),  # [n_targets, n_responses, n_trials, n_timepoints]
        "response_names": np.array(response_names),  # [n_responses]
        "target_trial_currents": np.full((n_targets, n_trials), np.nan),  # [n_targets, n_trials]
        "response_trial_coords": np.array(
            [mni_coords_dict[r] for r in response_names]
        ),  # [n_responses, 3]
    }

    for stim_i, stim in tqdm(enumerate(unique_stims), total=n_stims):
        logger.info(f"Working on stim electrode - {stim}")
        stim_paths = [path for path in pulse_paths if f"_{stim}_" in path.name]
        dists_stim_to_targets = np.array(
            [calc_euc_distance(mni_coords_dict, stim, t) for t in targets]
        )

        trial_idx = -1
        for pulse_path in stim_paths:

            # Load each pulse file
            mat_data = loadmat(pulse_path, data=["labels", "data"])

            pulse = normalize_pulse(mat_data, pulse_path)
            if pulse is False:
                logger.error(f"Skipping file due to normalization error: {pulse_path}")
                continue

            labels = np.array(mat_data["labels"])
            current = float(pulse_path.name.split("_")[2].strip("mA"))

            # Error if not consistent with prior assumptions
            if pulse.shape[1] != n_timepoints:
                logger.error(
                    f"Inconsistent number of timepoints in file {pulse_path}. Expected {n_timepoints}, got {pulse.shape[1]}."
                )
                continue
            elif not np.array_equal(labels, response_names):
                logger.error(
                    f"Inconsistent response names in file {pulse_path}. Expected {response_names}, got {labels}."
                )
                continue
            else:
                trial_idx += 1

            # Extract convergent data
            target_mask = np.isin(targets, labels)
            dist_mask = (
                dists_stim_to_targets >= dist_threshold
            )  # only consider targets greater than 20mm from stim
            selection_mask = target_mask & dist_mask
            label_indices = np.array([np.where(labels == t)[0][0] for t in targets[selection_mask]])
            convergent["data"][selection_mask, stim_i, trial_idx, :] = pulse[label_indices, :]
            convergent["stim_trial_currents"][stim_i, trial_idx] = current

            # Extract divergent data
            if stim in targets:
                dists_stim_to_responses = np.array(
                    [calc_euc_distance(mni_coords_dict, stim, r) for r in response_names]
                )
                response_mask = (
                    dists_stim_to_responses >= dist_threshold
                )  # only consider responses greater than 20mm from stim
                target_idx = np.where(targets == stim)[0][0]
                divergent["data"][target_idx, response_mask, trial_idx, :] = pulse[response_mask, :]
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


def get_pulse_paths(subj):
    """
    Dummy implementation of get_pulse_paths function.
    In actual implementation, this function would return paths to .mat pulse files.
    """

    config = load_yaml()

    subj_path = get_subj_path(subj, path="subj_preproc_path")
    pulse_dir = subj_path / config["Paths"]["subj_pulse_dir"].replace("$SUBJ", subj)

    # Simulate finding .mat files
    pulse_paths = list(pulse_dir.glob("*.mat"))

    return pulse_paths


def normalize_pulse(pulse, pulse_path):
    """
    Normalize pulse data by z-scoring each channel based on its pre-stimulus baseline.
    Args:
        pulse (np.array): Pulse data of shape [n_channels, n_timepoints].
        pulse_path (Path): Path to the pulse .mat file (for logging purposes).

    Returns:
        np.array: Normalized pulse data of the same shape.
    """

    subj, stim, ma = pulse_path.name.split("_")[:3]

    config = load_yaml()
    baseline_dir = pulse_path.parent.parent / config["Paths"]["subj_pretrain_dir"].replace(
        "$SUBJ", subj
    )
    baseline_path = baseline_dir / f"{subj}_{stim}_pre_train_{ma}.mat"

    if not baseline_path.exists():
        logger.error(f"Baseline file not found for normalization: {baseline_path}")
        return False  # return False to indicate failure

    baseline = loadmat(baseline_path, data=["pre_train_1", "pre_train_2", "pre_train_3", "labels"])
    pretrain_monopolar = np.hstack(
        (baseline["pre_train_1"], baseline["pre_train_2"], baseline["pre_train_3"])
    )

    n_chans = pulse["data"].shape[0]
    pretrain = np.empty(shape=(n_chans, pretrain_monopolar.shape[1]))

    for i in range(n_chans):
        monopolar_idxs = [
            np.where(np.array(baseline["labels"]) == c)[0][0] for c in pulse["labels"][i].split("-")
        ]
        pretrain[i, :] = (
            pretrain_monopolar[monopolar_idxs[0], :] - pretrain_monopolar[monopolar_idxs[1], :]
        )

    pretrain_mean = np.mean(pretrain, axis=1, keepdims=True)
    pretrain_std = np.std(pretrain, axis=1, keepdims=True)
    # Z-score normalization
    normalized_pulse = (pulse["data"] - pretrain_mean) / pretrain_std

    return normalized_pulse
