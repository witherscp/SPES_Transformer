"""
Inject noise into conv and div branches before entering MLP. 

Usage:
    python src/analysis/test_pathway_importance.py --model-path experiments/best_model.pt --test-subject Epat31
"""

from pathlib import Path
from argparse import ArgumentParser
import yaml
from loguru import logger
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.utils import load_yaml, move_to_device
from src.datasets.seeg_dataset import SEEGDataset
from src.models.model import SEEGFusionModel
from src.training.train import get_subject_indices
from src.training.evaluate import compute_metrics


def inject_noise(embedding, noise_std=0.0):
    """
    Add Gaussian noise to embedding.

    Args:
        embedding: torch.Tensor of shape [B, embed_dim]
        noise_std: standard deviation of Gaussian noise

    Returns:
        noisy_embedding: torch.Tensor of same shape
    """
    if noise_std == 0.0:
        return embedding

    noise = torch.randn_like(embedding) * noise_std
    return embedding + noise


def evaluate_with_noise(
    model,
    dataloader,
    device,
    conv_noise_std=0.0,
    div_noise_std=0.0,
):
    """
    Evaluate model with noise injected into specified pathway(s).

    Args:
        model: trained SEEGFusionModel
        dataloader: test DataLoader
        device: torch device
        conv_noise_std: noise std for convergent pathway (0.0 = no noise)
        div_noise_std: noise std for divergent pathway (0.0 = no noise)

    Returns:
        metrics: dict with accuracy, precision, recall, f1, auc
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            # Forward pass through pathway-specific components
            x_conv = inputs["convergent"]
            x_div = inputs["divergent"]
            conv_padding_mask = inputs["convergent_mask"]
            div_padding_mask = inputs["divergent_mask"]
            conv_coords = inputs["convergent_coords"]
            div_coords = inputs["divergent_coords"]

            B, n_stims, n_trials, n_timepoints = x_conv.shape
            n_responses = x_div.shape[1]

            # Extract non-padded trials for MSResNet input
            resnet_conv_input = x_conv.reshape(B * n_stims * n_trials, 1, n_timepoints)
            resnet_conv_input = resnet_conv_input[~conv_padding_mask.reshape(B * n_stims * n_trials)]
            resnet_div_input = x_div.reshape(B * n_responses * n_trials, 1, n_timepoints)
            resnet_div_input = resnet_div_input[~div_padding_mask.reshape(B * n_responses * n_trials)]

            # Create embeddings through MSResNet
            resnet_conv_output = model.conv_msresnet(resnet_conv_input)
            resnet_div_output = model.div_msresnet(resnet_div_input)

            # Reshape back to [B, n_electrodes, n_trials, embed_dim]
            conv_embeddings = torch.zeros(
                (B * n_stims * n_trials, resnet_conv_output.shape[1]),
                device=resnet_conv_output.device,
                dtype=resnet_conv_output.dtype,
            )
            conv_embeddings[~conv_padding_mask.reshape(B * n_stims * n_trials)] = resnet_conv_output
            conv_embeddings = conv_embeddings.view(B, n_stims, n_trials, -1)

            div_embeddings = torch.zeros(
                (B * n_responses * n_trials, resnet_div_output.shape[1]),
                device=resnet_div_output.device,
                dtype=resnet_div_output.dtype,
            )
            div_embeddings[~div_padding_mask.reshape(B * n_responses * n_trials)] = resnet_div_output
            div_embeddings = div_embeddings.view(B, n_responses, n_trials, -1)

            # Create pathway embeddings through transformers
            conv_emb = model.conv_transformer(
                conv_embeddings, conv_coords, key_padding_mask=conv_padding_mask
            )
            div_emb = model.div_transformer(
                div_embeddings, div_coords, key_padding_mask=div_padding_mask
            )

            # **INJECT NOISE HERE** (before concatenation)
            conv_emb_noisy = inject_noise(conv_emb, conv_noise_std)
            div_emb_noisy = inject_noise(div_emb, div_noise_std)

            # Join embeddings and classify
            joint_emb = torch.cat([conv_emb_noisy, div_emb_noisy], dim=1)
            logits = model.classifier(joint_emb)

            # Collect predictions
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of positive class

    # Compute metrics
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

    return metrics


def run_pathway_analysis(
    model,
    dataloader,
    device,
    noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
    test_subject=None,
):
    """
    Run complete pathway importance analysis.

    Tests three conditions:
    1. Noise on convergent pathway only
    2. Noise on divergent pathway only
    3. Noise on both pathways (control)

    Args:
        model: trained SEEGFusionModel
        dataloader: test DataLoader
        device: torch device
        noise_levels: list of noise standard deviations to test
        test_subject: subject name (for logging)

    Returns:
        results_df: pandas DataFrame with all results
    """
    results = []

    logger.info(f"Testing pathway importance for subject: {test_subject}")
    logger.info(f"Noise levels: {noise_levels}")

    for noise_std in noise_levels:
        # 1. Noise on convergent pathway only
        logger.info(f"Testing convergent pathway with noise_std={noise_std}")
        metrics_conv = evaluate_with_noise(
            model, dataloader, device,
            conv_noise_std=noise_std,
            div_noise_std=0.0
        )
        results.append({
            "subject": test_subject,
            "condition": "convergent_noisy",
            "noise_std": noise_std,
            **metrics_conv
        })

        # 2. Noise on divergent pathway only
        logger.info(f"Testing divergent pathway with noise_std={noise_std}")
        metrics_div = evaluate_with_noise(
            model, dataloader, device,
            conv_noise_std=0.0,
            div_noise_std=noise_std
        )
        results.append({
            "subject": test_subject,
            "condition": "divergent_noisy",
            "noise_std": noise_std,
            **metrics_div
        })

        # 3. Noise on both pathways (control)
        logger.info(f"Testing both pathways with noise_std={noise_std}")
        metrics_both = evaluate_with_noise(
            model, dataloader, device,
            conv_noise_std=noise_std,
            div_noise_std=noise_std
        )
        results.append({
            "subject": test_subject,
            "condition": "both_noisy",
            "noise_std": noise_std,
            **metrics_both
        })

    results_df = pd.DataFrame(results)
    return results_df


def main(model_path, test_subjects=None, noise_levels=None, **kwargs):
    """
    Main function to run pathway importance analysis.

    Args:
        model_path: path to trained model checkpoint (.pt file)
        test_subjects: list of subjects to test on (None = all subjects)
        noise_levels: list of noise stds to test (None = default)
        **kwargs: config parameters
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    # Load dataset
    full_dataset = SEEGDataset(
        subjects=test_subjects,
        embed_dim=kwargs["Parameters"]["embed_dim"],
        data_dir="src/data"
    )

    # Determine which subjects to test
    if test_subjects is not None:
        subjects_to_test = [s for s in full_dataset.subjects if s in test_subjects]
    else:
        subjects_to_test = full_dataset.subjects

    logger.info(f"Testing on subjects: {subjects_to_test}")

    all_results = []

    for test_subj in subjects_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject: {test_subj}")
        logger.info(f"{'='*60}")

        # Create test dataloader for this subject
        test_idx = get_subject_indices(full_dataset, [test_subj])
        test_ds = Subset(full_dataset, test_idx)

        test_loader = DataLoader(
            test_ds,
            batch_size=kwargs["Parameters"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        # Auto-detect model path if using subject-specific models
        if model_path is None or model_path == "auto":
            # Try to find subject-specific model
            subject_model_path = Path(f"experiments/{test_subj}_model_Fusion_seed_1_final_epoch_6.pt")
            if subject_model_path.exists():
                actual_model_path = subject_model_path
                logger.info(f"Auto-detected model for {test_subj}: {actual_model_path}")
            else:
                logger.error(f"No model found for subject {test_subj}")
                continue
        else:
            actual_model_path = model_path

        # Load model
        logger.info(f"Loading model from: {actual_model_path}")
        model = SEEGFusionModel(
            embed_dim=kwargs["Parameters"]["embed_dim"],
            num_layers=kwargs["Parameters"]["num_layers"],
            n_heads=kwargs["Parameters"]["n_heads"],
            n_classes=2,
            device=device,
        )

        # Load weights (strict=False to handle model architecture changes)
        state_dict = torch.load(actual_model_path, map_location=device, weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        model.to(device)
        model.eval()  # Set to eval mode (frozen)

        logger.success(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Run pathway analysis
        results_df = run_pathway_analysis(
            model=model,
            dataloader=test_loader,
            device=device,
            noise_levels=noise_levels,
            test_subject=test_subj,
        )

        all_results.append(results_df)

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    # Save results
    results_dir = Path("../results/pathway_analysis")
    results_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"pathway_importance_{timestamp}.csv"
    final_results.to_csv(results_path, index=False)

    logger.success(f"\nResults saved to: {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*60)

    for subject in subjects_to_test:
        logger.info(f"\nSubject: {subject}")
        subj_df = final_results[final_results["subject"] == subject]

        # Compare performance degradation at highest noise level
        baseline = subj_df[(subj_df["condition"] == "convergent_noisy") & (subj_df["noise_std"] == 0.0)]
        baseline_acc = baseline["accuracy"].values[0]

        for condition in ["convergent_noisy", "divergent_noisy"]:
            cond_df = subj_df[subj_df["condition"] == condition]
            max_noise = cond_df[cond_df["noise_std"] == max(noise_levels)]

            if not max_noise.empty:
                acc_drop = baseline_acc - max_noise["accuracy"].values[0]
                logger.info(f"  {condition}: accuracy drop = {acc_drop:.4f} (baseline={baseline_acc:.4f}, noisy={max_noise['accuracy'].values[0]:.4f})")

    return final_results


if __name__ == "__main__":

    parser = ArgumentParser(description="Test pathway importance via noise injection.")
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default="auto",
        help="Path to trained model checkpoint (.pt file), or 'auto' to auto-detect subject-specific models (default: auto)",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        nargs="+",
        default=None,
        help="Optional: Specific subject(s) to test on (e.g., --test-subjects Epat31 Spat37)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Noise standard deviations to test (default: [0.0, 0.1, 0.2, 0.5, 1.0, 2.0])",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="../logs",
        help="Directory to save log files. Default is '../logs'.",
    )

    args = parser.parse_args()

    # Setup logging
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logdir / f"pathway_analysis_{timestamp}.log"

    logger.add(
        log_file,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        mode="w",
    )

    # Load config
    config = load_yaml()

    logger.info("Starting pathway importance analysis")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test subjects: {args.test_subjects}")
    logger.info(f"Noise levels: {args.noise_levels}")
    logger.info(f"\nConfig parameters:")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Run analysis
    results = main(
        model_path=args.model_path,
        test_subjects=args.test_subjects,
        noise_levels=args.noise_levels,
        **config
    )

    logger.success("\nPathway importance analysis complete!")
