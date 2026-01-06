import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from loguru import logger
import nibabel as nib
import mat73
from scipy.io import loadmat as lm
import re
import torch


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


def loadmat(file_path, data=["all"], verbose=False):
    """Load a .mat file, handling both v7.3 and earlier versions.

    Args:
        file_path (str or Path): Path to the .mat file.
        data (list): List of data names to load. Default is ['all'].
            ['all'] loads all available data.
            ['labels', 'fs', 'pulse'] for example will load only those variables.
        verbose (bool): If True, print information about loaded data. Default is False.

    Returns:
        dict: A dictionary containing the contents of the .mat file.
    """

    outputs = {}

    # load .mat file
    try:
        mat = lm(file_path)
    except NotImplementedError:
        mat = mat73.loadmat(file_path)

    if data == ["all"]:
        data = [key for key in mat.keys() if not key.startswith("__")]

    for d in data:
        match d:
            case fs if fs in ["fs", "sfreq"]:
                outputs["fs"] = np.uint16(mat[fs])
                if isinstance(outputs["fs"], np.ndarray):
                    outputs["fs"] = outputs["fs"][0][0]

            case label if label in ["labels", "bip_montage_label"]:
                if isinstance(mat[label], list):
                    outputs[label] = mat[label]
                else:
                    if mat[label].shape[0] == 1:
                        outputs[label] = [l[0] for l in mat[label][0]]
                    elif mat[label].ndim == 1:
                        outputs[label] = mat[label]
                    else:
                        outputs[label] = [l[0][0] for l in mat[label]]

                # remove white spaces from labels
                outputs[label] = [l.replace(" ", "") for l in outputs[label]]  # get rid of spaces

            case s if re.fullmatch(r"pre_train_(\d{1,2})", s):
                outputs[s] = mat[s]

            case timeseries if timeseries in ["pulse", "filt_data", "data"]:
                if timeseries == "data":
                    for test_key in ["pulse", "filt_data"]:
                        if test_key in mat.keys():
                            outputs["data"] = mat[test_key]
                else:
                    outputs["data"] = mat[timeseries]

            case "filt_params":
                outputs["filt_params"] = mat["filt_params"][0]
            case _:
                if verbose:
                    logger.warning(f"Data '{d}' has no matching case and was not loaded.")
                continue

    for k, v in outputs.items():
        message = f"{k}: {type(v)}; "
        if isinstance(v, np.ndarray):
            message += f"shape: {v.shape}"
        else:
            message += f"value: {v}"
        if verbose:
            logger.info(message)

    return outputs


def load_yaml(yaml_f="src/config/default.yaml"):
    """Load a YAML configuration file.

    Args:
            yaml_f (str or Path): Path to the YAML file. Can be absolute or relative to package root.

    Returns:
        dict: A dictionary containing the YAML file contents.
    """

    yaml_path = Path(yaml_f)

    # If path is not absolute, resolve it relative to the package root
    if not yaml_path.is_absolute():
        # Get the directory containing this utils.py file
        utils_dir = Path(__file__).parent
        # Go up to package root (SPES_Transformer/)
        package_root = utils_dir.parent
        yaml_path = package_root / yaml_f

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_subj_path(subj, path="subj_seeg_path"):
    """
    Load the subject's directory for a given patient.

    Args:
        subj (str): The name of the patient.
        path (str): The key in the config file specifying the desired path type.

    Returns:
        Path: The subject's directory.
    """

    config = load_yaml()

    subj_dir = Path(config["paths"][path].replace("$SUBJ", subj))

    if subj_dir.is_dir() == False:
        possible_dirs = list(subj_dir.parent.glob(f"{subj}*"))
        if len(possible_dirs) == 1:
            subj_dir = possible_dirs[0]
            logger.warning(f"Using {subj_dir} for patient {subj}")
        else:
            logger.error(
                f"No {subj_dir} found for patient {subj} and possible directories: {possible_dirs}"
            )

    return subj_dir


def load_roi_assignments(subj, subj_seeg_dir):
    """
    Load the ROI (Region of Interest) assignments for a given patient.

    Args:
        subj (str): The name of the patient.
        subj_seeg_dir (Path): The subject's SEEG directory from get_subj_path(subj).

    Returns:
        pd.DataFrame: A DataFrame containing the ROI assignments, or an empty DataFrame if not found.
    """

    config = load_yaml()
    roi_df_path1 = subj_seeg_dir / config["paths"]["roi_assignments_file"]
    try:
        roi_df = pd.read_csv(roi_df_path1)
    except FileNotFoundError:
        roi_df_path2 = roi_df_path1.parent / ("SEEG_BipMontage_ROI_Assignments_EditedWithEEG.csv")
        try:
            roi_df = pd.read_csv(roi_df_path2)
        except FileNotFoundError:
            roi_df_path3 = roi_df_path2.parent / ("SEEG_BipMontage_ROI_Assignments.csv")
            try:
                roi_df = pd.read_csv(roi_df_path3)
            except FileNotFoundError:
                logger.error(
                    f"Could not find ROI assignments file for {subj} at {roi_df_path1}, {roi_df_path2}, or {roi_df_path3}."
                )
                return pd.DataFrame()

    roi_df["DKS Region Name"] = roi_df["DKS Region Name"].fillna("Unknown")

    return roi_df


@lru_cache(maxsize=None)
@logger.catch
def get_dks_region_dict(subj):
    """
    Get a dictionary mapping channel names (ex: LTP1-LTP2) to DKS region names for a given patient.

    Args:
        subj (str): Patient identifier to load the appropriate configuration and data files.

    Returns:
        dict: Mapping of channel names to DKS region names for given patient
    """

    subj_seeg_dir = get_subj_path(subj, path="subj_seeg_path")

    # ---------------- Load ROI assignments ----------------
    roi_df = load_roi_assignments(subj, subj_seeg_dir)
    if roi_df.empty:
        return {}

    # ---------------- Load Bipolar lookup table ----------------
    bip_lut = _load_bip_lut(subj, subj_seeg_dir)

    # ---------------- Add short labels to ROI df ----------------
    roi_df = _add_short_label_to_roi_df(roi_df, bip_lut)

    # ---------------- Build mapping ----------------
    pat_mapping_dict = dict(zip(roi_df["Short Bip Label"], roi_df["DKS Region Name"]))

    logger.success(
        f"Built DKS region dict for {subj} with {len(pat_mapping_dict)} channels (cached)"
    )
    return pat_mapping_dict


@lru_cache(maxsize=None)
@logger.catch
def get_dks_broad_region_dict(subj):
    """
    Get a dictionary mapping channel names (ex: LTP1-LTP2) to broader brain region categories for a given patient.

    Args:
        subj (str): Patient identifier to load the appropriate configuration and data files.

    Returns:
        dict: Mapping of channel names to broader brain region categories for given patient
    """
    dks_region_dict = get_dks_region_dict(subj)
    broad_region_dict = {
        chan: dks_to_broad_region(region) for chan, region in dks_region_dict.items()
    }
    logger.success(
        f"Built broad region dict for {subj} with {len(broad_region_dict)} channels (cached)"
    )
    return broad_region_dict


def dks_to_broad_region(dks_region_inp):
    """
    Maps a DKS region name to a broader brain region category.

    Args:
        dks_region_inp (str): The DKS region name to be mapped.

    Returns:
        str: The broader brain region category corresponding to the input DKS region.
    """
    dks_region = dks_region_inp.lower()

    if any(
        keyword in dks_region
        for keyword in [
            "caudalmiddlefrontal",
            "rostralmiddlefrontal",
            "superiorfrontal",
            "precentral",
            "paracentral",
            "frontalpole",
            "lateralorbitofrontal",
            "medialorbitofrontal",
            "" "parsopercularis",
            "parsorbitalis",
            "parstriangularis",
        ]
    ):
        return "Frontal"
    elif any(
        keyword in dks_region
        for keyword in [
            "inferiorparietal",
            "superiorparietal",
            "supramarginal",
            "precuneus",
            "postcentral",
        ]
    ):
        return "Parietal"
    elif any(
        keyword in dks_region
        for keyword in [
            "bankssts",
            "inferiortemporal",
            "middletemporal",
            "superiortemporal",
            "temporalpole",
            "transversetemporal",
            "fusiform",
            "entorhinal",
            "parahippocampal",
        ]
    ):
        return "Temporal"
    elif any(
        keyword in dks_region
        for keyword in ["cuneus", "lingual", "pericalcarine", "lateraloccipital"]
    ):
        return "Occipital"
    elif any(
        keyword in dks_region
        for keyword in [
            "caudalanteriorcingulate",
            "rostralanteriorcingulate",
            "isthmuscingulate",
            "posteriorcingulate",
        ]
    ):
        return "Cingulate"
    elif "insula" in dks_region:
        return "Insula"
    elif "amygdala" in dks_region:
        return "Amygdala"
    elif "hippocampus" in dks_region:
        return "Hippocampus"
    elif "thalamus" in dks_region:
        return "Thalamus"
    elif any(keyword in dks_region for keyword in ["pallidum", "putamen", "caudate", "accumbens"]):
        return "Basal Ganglia"
    elif "unknown" in dks_region:
        return "Unknown"
    elif "cerebellum" in dks_region:
        return "Cerebellum"
    else:
        logger.error(
            f"DKS region '{dks_region_inp}' not recognized. Please update the mapping function or exclude this region."
        )
        return "Error"


@lru_cache(maxsize=None)
@logger.catch
def get_soz_label_dict(subj, IZ_as_NIZ=True):
    """
    Get a dictionary mapping channel names (ex: LTP1-LTP2) to their corresponding SOZ labels for a given patient.

    Args:
        subj (str): Patient identifier to locate the appropriate stimulus files.
        IZ_as_NIZ (bool): If True, reclassify 'IZ' labels as 'NIZ'. Default is True.

    Returns:
        dict: A dictionary mapping channel names to their corresponding SOZ labels.
    """
    # load config
    config = load_yaml()

    soz_labels_df = pd.read_csv(
        config["paths"]["soz_labels_fpath"], names=["Patient", "Bipole", "Label"]
    )
    pat_labels = soz_labels_df[soz_labels_df.Patient == subj].copy()
    if pat_labels.empty:
        logger.error(f"{subj} not found in SOZ labels file: {config['paths']['soz_labels_fpath']}")
        return {}

    pat_labels["Bipole"] = pat_labels["Bipole"].str.replace(" ", "")

    # Map numeric labels to string labels
    pat_labels["Label"] = pat_labels["Label"].apply(map_label)

    if IZ_as_NIZ:
        pat_labels.loc[pat_labels.Label == "IZ", "Label"] = "NIZ"

    soz_label_dict = pat_labels.set_index("Bipole")["Label"].to_dict()

    logger.success(f"Built SOZ label dict for {subj} with {len(soz_label_dict)} channels (cached)")

    return soz_label_dict


def map_label(label):
    """Map numeric SOZ label to string label.

    Args:
        label (int): Numeric SOZ label (0, 1, 2, or 3).

    Returns:
        str: Corresponding string SOZ label ("NIZ", "SOZ", "PZ", or "IZ").
    """

    label = int(label)
    match label:
        case 0:
            return "NIZ"
        case 1:
            return "SOZ"
        case 2:
            return "PZ"
        case 3:
            return "IZ"


@lru_cache(maxsize=None)
@logger.catch
def get_mni_coords_dict(subj):
    """
    Load the MNI electrode coordinates for a given patient.

    Args:
        subj (str): Patient identifier to locate the appropriate coordinate file.

    Returns:
        dict: A dictionary mapping channel names to their MNI coordinates as (x, y, z) tuples.
    """

    coords_df = _get_patient_coords(subj)
    if coords_df.empty:
        return {}

    mni_affine = _load_mni_affine(subj)
    if mni_affine is None:
        return {}

    coords_pat = coords_df[["X", "Y", "Z"]].to_numpy()
    coords_mni = _transform_to_mni(coords_pat, mni_affine)

    contacts = coords_df["Contact"].to_numpy()
    mni_coords_dict = {c: coords_mni[i] for i, c in enumerate(contacts)}

    for coords in mni_coords_dict.values():
        if np.any(np.isnan(coords)):
            logger.error(
                "NaN coordinates are present. Problem with get_mni_coords_dict() is present."
            )
            return {}

    logger.success(
        f"Built MNI coords dict for {subj} with {len(mni_coords_dict)} channels (cached)"
    )

    return mni_coords_dict


def calc_euc_distance(mni_coords_dict, chan1, chan2):
    """Calculate the Euclidean distance between two channels.

    Args:
        mni_coords_dict (dict): A dictionary mapping channel names to their MNI coordinates as (x, y, z) tuples.
        chan1 (str): The name of the first channel.
        chan2 (str): The name of the second channel.

    Returns:
        float: The Euclidean distance between the two channels.
    """

    if chan1 not in mni_coords_dict.keys():
        logger.error(f"Channel {chan1} not found in coordinates DataFrame.")
        return np.nan

    if chan2 not in mni_coords_dict.keys():
        logger.error(f"Channel {chan2} not found in coordinates DataFrame.")
        return np.nan

    p1 = mni_coords_dict[chan1]
    p2 = mni_coords_dict[chan2]

    return np.linalg.norm(p1 - p2)


def _load_bip_lut(subj, subj_seeg_dir):
    """
    Load the bipolar lookup table for a given patient.

    Args:
        subj (str): The name of the patient.
        subj_seeg_dir (Path): The subject's SEEG directory from get_subj_path(subj).
    Returns:
        pd.DataFrame: A DataFrame containing the bipolar lookup table with columns ['Long', 'Short'].
    """
    config = load_yaml()

    bip_lut_path_cleaned = subj_seeg_dir / config["paths"]["subj_bip_lut_file"].replace(
        "$SUBJ", subj
    )
    bip_lut_path = subj_seeg_dir / bip_lut_path_cleaned.name.replace("_cleaned.csv", ".csv")

    try:
        bip_lut = pd.read_csv(bip_lut_path_cleaned, names=["Long", "Short"])
    except FileNotFoundError:
        bip_lut = pd.read_csv(bip_lut_path, names=["Long", "Short"])
    return bip_lut


def _add_short_label_to_roi_df(roi_df, bip_lut):
    """Add a 'Short Bip Label' column to the ROI DataFrame using the provided bipolar lookup table.

    Args:
        roi_df (pd.DataFrame): DataFrame containing ROI assignments with a 'Bip Label' column.
        bip_lut (pd.DataFrame): DataFrame containing bipolar lookup table with 'Long' and 'Short' columns.

    Returns:
        pd.DataFrame: Updated ROI DataFrame with an added 'Short Bip Label' column.
    """

    bip_lut = bip_lut.copy()

    bip_lut["Long_norm"] = (
        bip_lut["Long"].str.lower().str.replace(r"[\s\-]+", " ", regex=True).str.strip()
    )

    long_to_short = dict(zip(bip_lut["Long_norm"], bip_lut["Short"]))

    def shorten_electrode(label, long_to_short):
        """
        'ant cingulate 1'     -> 'AC1'
        'mid front gyrus 2 1' -> 'MFG2 1'
        """

        label_norm = re.sub(r"[\s\-]+", " ", label.lower()).strip()

        # extract trailing number
        m = re.search(r"(\d+)$", label_norm)
        if not m:
            raise ValueError(f"No contact number in '{label}'")

        contact = m.group(1)
        roi_text = label_norm[: m.start()].strip()

        for long_name, short in long_to_short.items():
            if roi_text == long_name:
                return f"{short}{contact}"

        raise ValueError(f"No ROI match for '{label}'")

    def shorten_bipole_with_contacts(bip_label, long_to_short):
        """
        Returns:
            anode, cathode, bip_short
        """

        if " - " not in bip_label:
            raise ValueError(f"Invalid bipole format: {bip_label}")

        left_raw, right_raw = bip_label.split(" - ", 1)

        anode = shorten_electrode(left_raw, long_to_short)
        cathode = shorten_electrode(right_raw, long_to_short)

        return anode, cathode, f"{anode}-{cathode}"

    roi_df[["Anode", "Cathode", "Short Bip Label"]] = roi_df["Bip Label"].apply(
        lambda x: pd.Series(shorten_bipole_with_contacts(x, long_to_short))
    )

    return roi_df


def _load_mni_affine(subj):
    """
    Load the MNI affine transformation matrix for a given patient.

    Args:
        subj (str): Patient identifier to locate the appropriate affine file.

    Returns:
        np.ndarray: The MNI affine transformation matrix with shape (4, 4).

        The matrix looks like this:
        [[1, 0, 0, t1],
         [0, 1, 0, t2],
         [0, 0, 1, t3],
         [  0,   0,   0,  1]]

         where the upper-left 3x3 submatrix contains rotation and scaling components,
         and the last column [t1, t2, t3] represents translation components.

         In this case, the patient coordinates have already been registered to MNI voxel space,
         so the affine matrix only encodes translation to convert voxel indices to MNI coordinates.

         To use this matrix to transform a coordinate from patient space to MNI space,
         you can use the following formula:

         [x_mni, y_mni, z_mni, 1] = [x_pat, y_pat, z_pat, 1] @ mni_affine.T
    """
    config_analysis = load_yaml()
    subj_seeg_dir = get_subj_path(subj, path="subj_seeg_path")
    mni_file = subj_seeg_dir / config_analysis["paths"]["mni_file"]

    mni_img = nib.load(mni_file)
    mni_affine = mni_img.affine

    return mni_affine


def matstruct_to_dict(obj):
    """
    Recursively convert scipy.io.matlab.mat_struct objects to dicts.
    """
    if hasattr(obj, "_fieldnames"):
        return {field: matstruct_to_dict(getattr(obj, field)) for field in obj._fieldnames}
    elif isinstance(obj, dict):
        return {k: matstruct_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [matstruct_to_dict(v) for v in obj]
    else:
        return obj


def loadmat_spes(file_path, data=["all"], verbose=False):
    """Load a .mat file, handling both v7.3 and earlier versions.

    Args:
        file_path (str or Path): Path to the .mat file.
        data (list): List of data names to load. Default is ['all'].
            ['all'] loads all available data.
            ['labels', 'fs', 'pulse'] for example will load only those variables.
        verbose (bool): If True, print information about loaded data. Default is False.

    Returns:
        dict: A dictionary containing the contents of the .mat file.
    """

    outputs = {}

    # load .mat file
    mat = lm(file_path, squeeze_me=True, struct_as_record=False)
    mat = matstruct_to_dict(mat)

    if data == ["all"]:
        data = [key for key in mat.keys() if not key.startswith("__")]

    for d in data:
        match d:
            case v if v in [
                "fs",
                "stim_sequence",
                "stim_channel",
                "subj",
                "config_params",
                "blocks",
                "labels",
            ]:
                outputs[v] = mat[v]

            case _:
                if verbose:
                    logger.warning(f"Data '{d}' has no matching case and was not loaded.")
                continue

    return outputs


def load_spes_data(subj):
    """Load SPES data for a given subject.
    Args:
        subj (str): Subject identifier.
    Returns:
        dict: A dictionary containing SPES data for the subject.
            keys = stim channels
            values = dicts with keys = (trains, pulse times, data, labels, fs, current)
    """

    config = load_yaml()
    data_dir = get_subj_path(subj, path="subj_preproc_path")
    preproc_dir = data_dir / config["paths"]["subj_train_dir"].replace("$SUBJ", subj)

    master_dict = {}

    ## check for existence of folder and presence of files
    if not preproc_dir.is_dir():
        logger.error(f"{preproc_dir} does not exist. Cannot proceed...")
        return {}

    if len(list(preproc_dir.glob(f"{subj}_*.mat"))) == 0:
        logger.error("No .mat files are present in preproc_dir. Cannot proceed...")
        return {}

    # First, retrieve stim sequence from any file
    mat_dict = loadmat_spes(
        list(preproc_dir.glob(f"{subj}_*.mat"))[0],
        data=["stim_sequence", "labels", "fs", "config_params"],
    )
    stim_sequence = mat_dict["stim_sequence"]
    labels = mat_dict["labels"]
    fs = mat_dict["fs"]
    config_params = mat_dict["config_params"]

    master_dict["stim_sequence"] = stim_sequence
    master_dict["labels"] = labels
    master_dict["fs"] = fs
    master_dict["config_params"] = config_params

    tol_ms = config["parameters"]["tol_ms"]
    tol_length = int((tol_ms / 1000) * fs)

    # Iterate through all stim blocks
    stim_dict = {}
    stim_trains = {}
    for stim_block in stim_sequence:
        stim_chan = stim_block.split("_")[0]
        block = stim_block.split("_")[1]

        stim_fpath = list(
            preproc_dir.glob(f"{subj}_{stim_chan.replace(' ', '')}_padded_trains.mat")
        )[0]

        mat_dict = loadmat_spes(stim_fpath, ["blocks"])
        trains = mat_dict["blocks"][block]["train"]

        if isinstance(trains, int):
            bad_pulses = mat_dict["blocks"][block]["bad_pulses"].get(f"train{trains}", 0)
            bad_train = bad_pulses == -1

            if not bad_train:

                train_data = mat_dict["blocks"][block]["filt_data"]
                pulse_times = mat_dict["blocks"][block]["local_pulse_times_s"]
                first_pulse, last_pulse = pulse_times[0], pulse_times[-1]
                expected_length = fs * (
                    (last_pulse + 1 - first_pulse) + config_params["full_train_pad_s"] * 2
                )

                if (
                    expected_length - tol_length
                    < train_data.shape[1]
                    < expected_length + tol_length
                ):
                    stim_dict[f"{stim_block}_train{trains}"] = {
                        "data": train_data,
                        "pulse_times": pulse_times,
                        "current": mat_dict["blocks"][block]["current"],
                        "global_train_start_s": mat_dict["blocks"][block]["global_train_start_s"],
                    }
                    stim_trains.setdefault(stim_block, []).append(trains)
                else:
                    logger.warning(
                        f"Train {trains} in block {block} for subject {subj} and stim channel {stim_chan} has unexpected length "
                        f"{train_data.shape[1]} samples, expected ~{expected_length} samples +/- {tol_length} samples. Skipping."
                    )
        else:
            for i, train in enumerate(trains):

                bad_pulses = mat_dict["blocks"][block]["bad_pulses"].get(f"train{train}", 0)

                bad_train = False
                if isinstance(bad_pulses, int):
                    bad_train = bad_pulses == -1

                if not bad_train:

                    train_data = mat_dict["blocks"][block]["filt_data"][i]
                    pulse_times = mat_dict["blocks"][block]["local_pulse_times_s"][i]
                    first_pulse, last_pulse = pulse_times[0], pulse_times[-1]
                    expected_length = fs * (
                        (last_pulse + 1 - first_pulse) + config_params["full_train_pad_s"] * 2
                    )

                    if (
                        expected_length - tol_length
                        < train_data.shape[1]
                        < expected_length + tol_length
                    ):
                        stim_dict[f"{stim_block}_train{train}"] = {
                            "data": train_data,
                            "pulse_times": pulse_times,
                            "current": mat_dict["blocks"][block]["current"][i],
                            "global_train_start_s": mat_dict["blocks"][block][
                                "global_train_start_s"
                            ][i],
                        }
                        stim_trains.setdefault(stim_block, []).append(train)
                    else:
                        logger.warning(
                            f"Train {train} in block {block} for subject {subj} and stim channel {stim_chan} has unexpected length "
                            f"{train_data.shape[1]} samples, expected ~{expected_length} samples +/- {tol_length} samples. Skipping."
                        )

    master_dict["stim_data"] = stim_dict
    master_dict["stim_trains"] = stim_trains

    return master_dict


def _transform_to_mni(coords_pat, mni_affine):
    """
    Transform registered coordinates from MNI voxel space to MNI coordinate space using the provided affine transformation matrix.

    Args:
        coords_pat (np.ndarray): An array of shape (N, 3) containing N registered coordinates in voxel space.
        mni_affine (np.ndarray): The MNI affine transformation matrix with shape (4, 4).

    Returns:
        np.ndarray: An array of shape (N, 3) containing the transformed coordinates in MNI coordinate space.
    """
    # Add a column of ones to the patient coordinates to make them homogeneous
    coords_pat_hom = np.hstack([coords_pat, np.ones((coords_pat.shape[0], 1))])
    # Apply the affine transformation
    coords_mni_hom = coords_pat_hom @ mni_affine.T
    # Convert back to Cartesian coordinates
    coords_mni = coords_mni_hom[:, :3]

    return coords_mni


def _split_contacts_safe(bip_label):
    """Splits a bipolar long label into its two constituent contacts.
    Use in a pandas .apply() to create two new columns."""

    parts = bip_label.split("-")
    if len(parts) == 2:
        return pd.Series(parts)
    elif len(parts) > 2 & len(parts) % 2 == 0:
        # Assume hyphens are in electrode names, e.g. "R-hippo1-R-hippo2"
        mid = len(parts) // 2
        contact1 = "-".join(parts[:mid])
        contact2 = "-".join(parts[mid:])
        return pd.Series([contact1, contact2])
    else:
        logger.warning(f"Could not split bipole label {bip_label}. Returning NaN.")
        return pd.Series([np.nan, np.nan])


def _get_patient_coords(subj):
    """
    Load the electrode coordinates for a given patient. Coordinates have been registered to MNI voxel space.

    Args:
        subj (str): Patient identifier to locate the appropriate coordinate file.
    Returns:
        pd.DataFrame: A DataFrame containing electrode coordinates with columns ['Contact', 'X', 'Y', 'Z'].
    """

    subj_seeg_dir = get_subj_path(subj, path="subj_seeg_path")

    roi_df = load_roi_assignments(subj, subj_seeg_dir)
    if roi_df.empty:
        return pd.DataFrame()

    bip_lut = _load_bip_lut(subj, subj_seeg_dir)
    roi_df = _add_short_label_to_roi_df(roi_df, bip_lut)

    coords_df = _load_pat_coords(subj, subj_seeg_dir, bip_lut)
    if coords_df.empty:
        return pd.DataFrame()

    bipole_coords_df = _compute_bipole_coords(coords_df, roi_df)

    return bipole_coords_df


def _compute_bipole_coords(coords_df, roi_df):
    """Compute the bipole coordinates from the given coordinates and ROI DataFrames.

    Args:
        coords_df (pd.DataFrame): DataFrame containing electrode coordinates.
        roi_df (pd.DataFrame): DataFrame containing ROI information.

    Returns:
        pd.DataFrame: DataFrame containing bipole coordinates.
    """

    # Step 1: Merge coords for contact1 and contact2
    merged = (
        roi_df.merge(coords_df, left_on="Anode", right_on="Contact")
        .rename(columns={"X": "X1", "Y": "Y1", "Z": "Z1"})
        .drop(columns="Contact")
        .merge(coords_df, left_on="Cathode", right_on="Contact")
        .rename(columns={"X": "X2", "Y": "Y2", "Z": "Z2"})
        .drop(columns="Contact")
    )

    # Step 2: Compute average coordinates
    merged["X"] = merged[["X1", "X2"]].mean(axis=1)
    merged["Y"] = merged[["Y1", "Y2"]].mean(axis=1)
    merged["Z"] = merged[["Z1", "Z2"]].mean(axis=1)

    # Step 3: Combine into final dataframe
    bipole_coords_df = pd.DataFrame(
        {
            "Contact": roi_df["Short Bip Label"],
            "X": merged["X"],
            "Y": merged["Y"],
            "Z": merged["Z"],
        }
    )

    return bipole_coords_df


def _load_pat_coords(subj, subj_seeg_dir, bip_lut):
    """Load the electrode coordinates for a given patient (DO NOT call this function directly, use get_patient_coords).

    Args:
        subj (str): The name of the patient.
        subj_seeg_dir (Path): The subject's SEEG directory.

    Returns:
        pd.DataFrame: A DataFrame containing electrode coordinates with columns ['Contact', 'X', 'Y', 'Z'].
    """

    config = load_yaml()

    coords_file = subj_seeg_dir / config["paths"]["subj_coords_file"].replace("$SUBJ", subj)
    if coords_file.is_file():
        coords = np.loadtxt(coords_file, skiprows=1)
    else:
        glob_search = list(subj_seeg_dir.glob(f"{subj}_*_coords_registered.csv"))
        if len(glob_search) == 1:
            coords_file = glob_search[0]
            coords = np.loadtxt(coords_file, skiprows=1)
        else:
            logger.error(f"Could not find coords_registered.csv file for {subj}")
            return pd.DataFrame()

    contact_file = Path(str(coords_file).replace("coords_registered", "raw_contact_names"))
    contacts = pd.read_csv(contact_file, header=None, names=["Long", "Num"])
    contacts["Short"] = contacts["Long"].map(dict(zip(bip_lut.Long, bip_lut.Short)))
    contacts["Contact"] = contacts["Short"] + contacts["Num"].astype(str)

    coords_df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    coords_df["Contact"] = contacts["Contact"].values

    return coords_df
