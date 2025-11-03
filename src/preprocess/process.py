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
)


@logger.catch
def build_subject_pt(subj):
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

    # # get mni coords dict
    # mni_coords_dict = get_mni_coords_dict(subj)

    ## ------ Load and format pulse data ------

    # Get paths to all .mat pulse files
    pulse_paths = get_pulse_paths(subj)

    # Load convergent and divergent datasets
    data_dict = combine_pulses(
        pulse_paths=pulse_paths,
        possible_targets=soz_label_dict.keys(),  # most restrictive set of electrodes (must know for electrode to be target)
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
        },
        "divergent": {
            "data": torch.tensor(data_dict["divergent"]["data"], dtype=torch.float32),
            "response_names": data_dict["divergent"]["response_names"],
            "target_trial_currents": torch.tensor(
                data_dict["divergent"]["target_trial_currents"], dtype=torch.float32
            ),
        },
        "target_labels": torch.tensor(target_labels, dtype=torch.long),
    }

    data_dir = Path("../../data/processed")
    torch.save(data, data_dir / f"{subj}.pt")
    return True


def combine_pulses(pulse_paths, possible_targets):
    """
    Combine convergent and divergent pulse data from multiple .mat files.
    Args:
        pulse_paths (list of str): List of paths to .mat pulse files.
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
    mat = loadmat(pulse_paths[0])
    n_timepoints = mat["pulse"].shape[1]
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
        "stim_trial_currents": np.full((n_stims, n_trials), np.nan),  # [n_stims, n_trials]
    }

    divergent = {
        "data": np.full(
            (n_targets, n_responses, n_trials, n_timepoints), np.nan
        ),  # [n_targets, n_responses, n_trials, n_timepoints]
        "response_names": np.array(response_names),  # [n_responses]
        "target_trial_currents": np.full((n_targets, n_trials), np.nan),  # [n_targets, n_trials]
    }

    for stim_i, stim in tqdm(enumerate(unique_stims), total=n_stims):
        logger.info(f"Working on stim electrode - {stim}")
        stim_paths = [path for path in pulse_paths if f"_{stim}_" in path.name]

        trial_idx = -1
        for pulse_path in stim_paths:

            # Load each pulse file
            mat_data = loadmat(pulse_path)
            pulse = mat_data["pulse"]
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
            target_indices = np.isin(targets, labels)
            label_indices = np.array([np.where(labels == t)[0][0] for t in targets if t in labels])
            convergent["data"][target_indices, stim_i, trial_idx, :] = pulse[label_indices, :]
            convergent["stim_trial_currents"][stim_i, trial_idx] = current

            # Extract divergent data
            if stim in targets:
                target_idx = np.where(targets == stim)[0][0]
                divergent["data"][target_idx, :, trial_idx, :] = pulse
                divergent["target_trial_currents"][target_idx, trial_idx] = current

    # Check for nan entries which suggests error in logic
    if np.sum(np.isnan(convergent["data"][:, :, 0, 0])):
        logger.error(
            "Something went wrong filling in convergent array. Check logic in debugging mode!"
        )
    elif np.sum(np.isnan(divergent["data"][:, :, 0, 0])):
        logger.error(
            "Something went wrong filling in divergent array. Check logic in debugging mode!"
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
