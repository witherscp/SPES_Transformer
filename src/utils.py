import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from loguru import logger
import nibabel as nib
import mat73
from scipy.io import loadmat as lm
import torch


def loadmat(file_path):
    """Load a .mat file, handling both v7.3 and earlier versions.

    Args:
        file_path (str or Path): Path to the .mat file.

    Returns:
        dict: A dictionary containing the contents of the .mat file.
    """

    outputs = {}
    try:
        spes = lm(file_path)
        outputs["fs"] = np.uint16(spes["fs"][0][0])
        if spes["labels"].shape[0] == 1:
            outputs["labels"] = [l[0] for l in spes["labels"][0]]
        elif spes["labels"].ndim == 1:
            outputs["labels"] = spes["labels"]
        else:
            outputs["labels"] = [l[0][0] for l in spes["labels"]]

    except NotImplementedError:
        spes = mat73.loadmat(file_path)
        outputs["fs"] = np.uint16(spes["fs"])
        outputs["labels"] = spes["labels"]

    try:
        outputs["pulse"] = spes["filt_data"]
    except KeyError:
        outputs["pulse"] = spes["pulse"]

    outputs["labels"] = [l.replace(" ", "") for l in outputs["labels"]]  # get rid of spaces

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

    subj_dir = Path(config["Paths"][path].replace("$SUBJ", subj))

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
        subj_seeg_dir (Path): The subject's SEEG directory from get_subj_path(subj, path='subj_seeg_path').

    Returns:
        pd.DataFrame: A DataFrame containing the ROI assignments, or an empty DataFrame if not found.
    """

    config = load_yaml()
    roi_df_path = subj_seeg_dir / config["Paths"]["roi_assignments_file"]

    try:
        roi_df = pd.read_csv(roi_df_path)
    except FileNotFoundError:
        roi_df_path = roi_df_path.parent / ("SEEG_BipMontage_ROI_Assignments.csv")
        try:
            roi_df = pd.read_csv(roi_df_path)
        except FileNotFoundError:
            logger.error(
                f"Could not find ROI assignments file for {subj} at {roi_df_path} or {config['paths']['roi_assignments_file']}"
            )
            return pd.DataFrame()

    roi_df["DKS Region Name"] = roi_df["DKS Region Name"].fillna("Unknown")
    roi_df["Bip Label"] = roi_df["Bip Label"].str.replace(" ", "")

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
        config["Paths"]["soz_labels_fpath"], names=["Patient", "Bipole", "Label"]
    )
    pat_labels = soz_labels_df[soz_labels_df.Patient == subj].copy()
    if pat_labels.empty:
        logger.error(f"{subj} not found in SOZ labels file: {config['Paths']['soz_labels_fpath']}")
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

    logger.success(
        f"Built MNI coords dict for {subj} with {len(mni_coords_dict)} channels (cached)"
    )

    return mni_coords_dict


def calc_euc_distance(subj, chan1, chan2):
    """Calculate the Euclidean distance between two channels.

    Args:
        subj (str): The subject identifier.
        chan1 (str): The name of the first channel.
        chan2 (str): The name of the second channel.

    Returns:
        float: The Euclidean distance between the two channels.
    """

    mni_coords_dict = get_mni_coords_dict(subj)

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
        subj_seeg_dir (Path): The subject's SEEG directory from get_subj_path(subj, path='subj_seeg_path').

    Returns:
        pd.DataFrame: A DataFrame containing the bipolar lookup table with columns ['Long', 'Short'].
    """
    config = load_yaml()

    bip_lut_path = subj_seeg_dir / config["Paths"]["subj_bip_lut_file"].replace("$SUBJ", subj)
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

    bip_lut.Short = bip_lut.Short.str.replace(" ", "")
    bip_lut.Long = bip_lut.Long.str.replace(" ", "")
    long_to_short = dict(zip(bip_lut.Long, bip_lut.Short))

    # Split long bipoles (e.g., "R hippo 1 - R hippo 2") into individual contacts
    bip_splits = roi_df["Bip Label"].apply(_split_contacts_safe)
    bip_splits.columns = ["contact1", "contact2"]

    # Extract prefix and numeric parts
    for c in ["contact1", "contact2"]:
        roi_df[f"{c}_prefix"] = bip_splits[c].str.extract(r"(.+?)(\d+)$")[0]
        roi_df[f"{c}_num"] = bip_splits[c].str.extract(r"(.+?)(\d+)$")[1]

        # Map long → short electrode name
        roi_df[f"{c}_short_prefix"] = roi_df[f"{c}_prefix"].map(long_to_short)
        roi_df[f"{c}_short_prefix"] = roi_df[f"{c}_short_prefix"].fillna(roi_df[f"{c}_prefix"])
        roi_df[f"{c}_short"] = roi_df[f"{c}_short_prefix"] + roi_df[f"{c}_num"]

    # Combine into final short-form bipole
    roi_df["Short Bip Label"] = roi_df["contact1_short"] + "-" + roi_df["contact2_short"]

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
    mni_file = subj_seeg_dir / config_analysis["Paths"]["mni_file"]

    mni_img = nib.load(mni_file)
    mni_affine = mni_img.affine

    return mni_affine


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
        roi_df.merge(coords_df, left_on="contact1_short", right_on="Contact")
        .rename(columns={"X": "X1", "Y": "Y1", "Z": "Z1"})
        .drop(columns="Contact")
        .merge(coords_df, left_on="contact2_short", right_on="Contact")
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

    coords_file = subj_seeg_dir / config["Paths"]["subj_coords_file"].replace("$SUBJ", subj)
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
    contacts["Long"] = contacts["Long"].str.replace(" ", "")
    contacts["Short"] = contacts["Long"].map(dict(zip(bip_lut.Long, bip_lut.Short)))
    contacts["Contact"] = contacts["Short"] + contacts["Num"].astype(str)

    coords_df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    coords_df["Contact"] = contacts["Contact"].values

    return coords_df
