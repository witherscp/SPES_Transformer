from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.preprocess.process import build_subject_pt


def main():

    parser = ArgumentParser(
        description="Prepare .pt file for a given subject from their .mat files present in "
        "subj_pulse_dir as specified in default.yaml."
    )
    parser.add_argument(
        "-s",
        "--subj",
        type=str,
        help="Subject name in the format '*pat*' (e.g. Epat26). ",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="../logs",
        help="Directory to save log files. Default is '../logs'.",
    )

    args = parser.parse_args()
    subj = args.subj
    logdir = Path(args.logdir)

    # Create log directory if it doesn't exist
    logdir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logdir / f"process_subject_{subj}_{timestamp}.log"

    # Configure logger with process ID in format
    logger.add(
        log_file,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | PID:{process} | {level} | {message}",
        mode="w",
    )

    logger.info(f"Creating .pt file for subject: {subj}")
    logger.info(f"Parameters in default.yaml are set as follows:")
    logger.info(open("../config/default.yaml").read())

    success = build_subject_pt(subj)

    if success:
        logger.success(f".pt file for subject {subj} created successfully.")
    else:
        logger.error(f"Failed to create .pt file for subject {subj}.")


if __name__ == "__main__":
    main()
