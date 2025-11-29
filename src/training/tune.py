import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import FixedTrial
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any, List
from argparse import ArgumentParser
from loguru import logger
from datetime import datetime
from math import ceil

from src.datasets.seeg_dataset import SEEGDataset
from src.models.model import SEEGFusionModel
from src.training.train import get_subject_indices, compute_class_weights, move_to_device
from src.utils import load_yaml


def suggest_from_cfg(trial: optuna.trial.Trial, name: str, param_cfg: Dict[str, Any]):
    ptype = param_cfg["type"]
    if ptype == "int":
        if "step" in param_cfg:
            return trial.suggest_int(
                name, param_cfg["low"], param_cfg["high"], step=param_cfg["step"]
            )
        return trial.suggest_int(name, param_cfg["low"], param_cfg["high"])
    if ptype == "categorical":
        return trial.suggest_categorical(name, param_cfg["choices"])
    if ptype == "loguniform":
        return trial.suggest_float(name, param_cfg["low"], param_cfg["high"], log=True)
    if ptype == "uniform":
        return trial.suggest_float(name, param_cfg["low"], param_cfg["high"])
    raise ValueError(f"Unknown param type: {ptype} for {name}")


def get_splits(subjects, val_ratio):

    # Create inner splits
    inner_splits = []
    # Split based on val_ratio
    n_val_subjs = max(int(len(subjects) * val_ratio), 1)
    n_splits = ceil(len(subjects) / n_val_subjs)

    for i in range(n_splits):
        if (i + 1) * n_val_subjs > len(subjects):
            val_subjs = subjects[i * n_val_subjs :]
        else:
            val_subjs = subjects[i * n_val_subjs : (i + 1) * n_val_subjs]
        train_subjs = [s for s in subjects if s not in val_subjs]
        inner_splits.append((train_subjs, val_subjs))

    return inner_splits


def objective_for_subjects(
    trial: optuna.trial.Trial,
    subjects: List[str],
    tune_cfg: Dict[str, Any],
    device: str,
) -> float:
    """
    Run K-fold proxy tuning where each subject in `subjects[:n_folds]` is used as validation once.
    Uses a monotonically increasing `report_step` counter to avoid Optuna duplicate-step warnings.
    Returns average best validation loss across folds (minimize).
    """

    # read config
    search = tune_cfg["search"]
    n_folds = tune_cfg.get("n_folds", len(subjects))
    n_epochs = tune_cfg["n_epochs"]
    batch_size = tune_cfg["batch_size"]
    patience = tune_cfg["patience"]

    # sample hyperparameters dynamically
    trial_params = {}
    for hp_name, hp_cfg in search.items():
        if hp_name != 'max_lr':
            trial_params[hp_name] = suggest_from_cfg(trial, hp_name, hp_cfg)
    trial.set_user_attr("trial_params", trial_params)

    trial_params['max_lr'] = trial.suggest_float(
        "max_lr",
        low=(base_lr + 1e-12),  # smallest possible margin above base_lr
        high=tune_cfg["search"]["max_lr"]["high"],
        log=tune_cfg["search"]["max_lr"]["type"] == "loguniform"
    )

    # extract hyperparameters
    embed_dim = trial_params["embed_dim"]
    num_layers = trial_params["num_layers"]
    n_heads = trial_params["n_heads"]
    weight_decay = trial_params["weight_decay"]
    base_lr = trial_params["base_lr"]
    max_lr = trial_params['max_lr']
    pct_start = trial_params["pct_start"]

    # Load dataset once per trial (with requested embed_dim)
    full_dataset = SEEGDataset(subjects=subjects, embed_dim=embed_dim, verbose=False)

    # Initialize report counter for this trial
    report_step = 0

    fold_best_losses = []

    fold_splits = get_splits(subjects, val_ratio=tune_cfg["val_ratio"])

    # iterate validation subjects (K-fold using provided list)
    folds_to_run = min(n_folds, len(fold_splits))
    for fold_idx in range(folds_to_run):
        train_subjs = fold_splits[fold_idx][0]
        val_subjs = fold_splits[fold_idx][1]

        train_idx = get_subject_indices(full_dataset, train_subjs)
        val_idx = get_subject_indices(full_dataset, val_subjs)

        if len(train_idx) == 0 or len(val_idx) == 0:
            # nothing to train/validate on this fold: prune the trial
            raise optuna.exceptions.TrialPruned(f"Empty fold for val_subjs={val_subjs}")

        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        }

        # Build model and optimizer/scheduler
        model = SEEGFusionModel(
            embed_dim=embed_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            n_classes=2,
            device=device,
        ).to(device)

        weights = compute_class_weights(train_ds)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=n_epochs,
            steps_per_epoch=len(dataloaders["train"]),
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        best_val_loss = float("inf")
        epochs_no_improve = 0

        # training loop
        for epoch in range(1, n_epochs + 1):
            model.train()
            for batch_inputs, batch_labels in dataloaders["train"]:
                batch_inputs = move_to_device(batch_inputs, device)
                batch_labels = move_to_device(batch_labels, device)

                optimizer.zero_grad()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_labels)

                # guard against NaN loss
                if torch.isnan(loss):
                    # skip this batch
                    continue

                loss.backward()

                optimizer.step()
                # step scheduler once per optimizer.step
                try:
                    scheduler.step()
                except Exception:
                    # scheduler could fail if steps_per_epoch wrong; ignore to keep trial running
                    logger.error(
                        "Scheduler failed to step. There must be a problem with steps_per_epoch and n_epochs."
                    )

            # validation after epoch
            model.eval()
            val_loss_total = 0.0
            val_count = 0
            with torch.no_grad():
                for val_inputs, val_labels in dataloaders["val"]:
                    val_inputs = move_to_device(val_inputs, device)
                    val_labels = move_to_device(val_labels, device)

                    outputs = model(val_inputs)
                    loss = criterion(outputs, val_labels)

                    # accumulate by number of samples (assumes field 'convergent' exists in inputs)
                    # fallback to batch size if not present
                    try:
                        n_samples = val_inputs["convergent"].size(0)
                    except Exception:
                        n_samples = val_labels.size(0)

                    val_loss_total += loss.item() * n_samples
                    val_count += n_samples

            epoch_val_loss = val_loss_total / val_count
            logger.info(f'Report step {report_step + 1}: val loss - {epoch_val_loss}')

            # Report the epoch metric to Optuna with a monotonic step counter
            report_step += 1
            trial.report(epoch_val_loss, report_step)

            # pruning check
            if trial.should_prune():
                # cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

            # early stopping per fold
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    break

        fold_best_losses.append(best_val_loss)

        # release model & free memory before next fold
        del model, optimizer, scheduler, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # final objective: average validation loss across folds (minimize)
    return float(np.mean(fold_best_losses))


def load_phase1(phase1_file, tune_cfg, out_dir):

    logger.info(f"phase1_file provided. Loading study and running Phase 2 only...")

    phase1_fpath = out_dir / phase1_file
    try:
        study = joblib.load(phase1_fpath)
        logger.success(f"Loaded Optuna study (phase 1) from {phase1_file}")
    except Exception as e:
        logger.error(f"Failed to load phase1_file: {e}")
        return

    # Extract completed trials from the loaded study
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not finished_trials:
        logger.warning("No completed trials found in study. Cannot run Phase 2.")
        return

    # Sort by best (lowest loss)
    finished_trials = sorted(finished_trials, key=lambda t: t.value)

    # Build Phase-1 leaderboard entries (top-K)

    top_k = tune_cfg["top_k"]
    top_trials = finished_trials[: min(len(finished_trials), top_k)]
    top_params_list = []
    for t in top_trials:
        # Prefer stored search-space suggestions from user_attrs, fallback to Optuna params
        params = t.user_attrs.get("trial_params", t.params)
        top_params_list.append({"trial_number": t.number, "value": t.value, "params": params})

    logger.success(f"Rebuilt Phase-1 top_params_list.")

    return top_params_list


def run_phase1(tune_cfg, out_dir, device, timestamp):

    subjects_phase1 = tune_cfg["subjects_phase1"]
    n_trials = tune_cfg["n_trials"]
    top_k = tune_cfg["top_k"]

    study = optuna.create_study(
        direction="minimize",
        pruner=SuccessiveHalvingPruner(
            min_resource=tune_cfg["min_resource"], reduction_factor=3, bootstrap_count=1
        ),
    )

    logger.info("Starting Phase 1 (small proxy) tuning...")
    study.optimize(
        lambda t: objective_for_subjects(t, subjects_phase1, tune_cfg, device),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    logger.success(f"Phase 1 complete. Best value: {study.best_value}")

    # save study
    study_path = out_dir / f"optuna_study_phase1_{timestamp}.pkl"
    try:
        joblib.dump(study, study_path)
        logger.success(f"Saved study to {study_path}")
    except Exception:
        logger.error("Could not save Optuna study (joblib not available)")

    # extract top_k trials (by value, lower is better)
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    finished_trials = sorted(finished_trials, key=lambda t: t.value)  # ascending
    top_trials = finished_trials[: min(len(finished_trials), top_k)]

    # write top_k params
    top_params_list = []
    for t in top_trials:
        params = t.user_attrs.get("trial_params", t.params)
        top_params_list.append({"trial_number": t.number, "value": t.value, "params": params})

    with open(out_dir / f"phase1_top_params_{timestamp}.yaml", "w") as f:
        yaml.safe_dump(top_params_list, f)

    logger.success(f"Phase 1 top-{len(top_params_list)} saved.")

    return top_params_list


def run_phase2(top_params_list, tune_cfg, out_dir, device, timestamp):

    subjects_phase2 = tune_cfg["subjects_phase2"]

    # Run Phase-2 evaluations on larger cohort
    logger.info("Starting Phase 2: evaluating loaded top-K candidates on larger cohort...")
    phase2_results = []
    for entry in top_params_list:
        params = entry["params"]
        fixed_trial = FixedTrial(params)

        try:
            # This call runs fresh Phase-2 evaluation; should not call trial.report() internally on old steps
            val = objective_for_subjects(fixed_trial, subjects_phase2, tune_cfg, device)
        except optuna.exceptions.TrialPruned:
            val = float("inf")
        except Exception as e:
            logger.error(f"Phase 2 eval crash on trial {entry['trial_number']}: {e}")
            val = float("inf")

        phase2_results.append({"params": params, "value_on_phase2": val})

    # Sort by Phase-2 performance
    phase2_results = sorted(phase2_results, key=lambda x: x["value_on_phase2"])

    with open(out_dir / f"phase2_results_{timestamp}.yaml", "w") as f:
        yaml.safe_dump(phase2_results, f)

    logger.success(f"Phase 2 complete. Results saved to phase2_results_{timestamp}.yaml")
    return


def main(timestamp, phase1_file=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tune_cfg = load_yaml("src/config/tune.yaml")

    out_dir = Path("../experiments")
    out_dir.mkdir(exist_ok=True, parents=True)

    if phase1_file:
        top_params_list = load_phase1(phase1_file, tune_cfg, out_dir)
    else:
        top_params_list = run_phase1(tune_cfg, out_dir, device, timestamp)

    run_phase2(top_params_list, tune_cfg, out_dir, device, timestamp)


if __name__ == "__main__":

    parser = ArgumentParser(description="Tune hyperparameters using Optuna in two-staged process.")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="../logs",
        help="Directory to save log files. Default is '../logs'.",
    )
    parser.add_argument(
        "--phase1_file",
        type=str,
        default=None,
        help="optional phase 1 .pkl file to skip directly to phase 2",
    )

    args = parser.parse_args()
    logdir = Path(args.logdir)
    phase1_file = args.phase1_file

    # Create log directory if it doesn't exist
    logdir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if phase1_file:
        timestamp = phase1_file.split("_")[-1]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logdir / f"tune_{timestamp}.log"

    # Configure logger with process ID in format
    logger.add(
        log_file,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | PID:{process} | {level} | {message}",
        mode="w",
    )

    logger.info(f"Parameters in tune.yaml are set as follows:")
    logger.info(open("../config/tune.yaml").read())

    main(timestamp, phase1_file)
