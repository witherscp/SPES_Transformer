"""
Visualization script for pathway importance analysis results.

Generates plots showing performance degradation as noise increases for each pathway.

Usage:
    python src/analysis/plot_pathway_results.py --results-file results/pathway_analysis/pathway_importance_20241205_123456.csv
"""

from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def plot_pathway_comparison(results_df, save_dir=None):
    """
    Create comprehensive visualization of pathway importance analysis.

    Args:
        results_df: DataFrame with pathway analysis results
        save_dir: directory to save plots (None = display only)
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    subjects = results_df["subject"].unique()
    metrics = ["accuracy", "auroc", "f1", "sensitivity", "specificity"]

    # 1. Plot for each subject separately
    for subject in subjects:
        subj_df = results_df[results_df["subject"] == subject]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Pathway Importance Analysis - {subject}", fontsize=16, fontweight="bold")

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            # Plot each condition
            for condition, label, marker in [
                ("convergent_noisy", "Convergent pathway noisy", "o"),
                ("divergent_noisy", "Divergent pathway noisy", "s"),
                ("both_noisy", "Both pathways noisy", "^"),
            ]:
                cond_df = subj_df[subj_df["condition"] == condition].sort_values("noise_std")
                ax.plot(
                    cond_df["noise_std"],
                    cond_df[metric],
                    marker=marker,
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

            ax.set_xlabel("Noise Standard Deviation", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f"{metric.capitalize()}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / f"pathway_comparison_{subject}.png", dpi=300, bbox_inches="tight")
            plt.savefig(save_dir / f"pathway_comparison_{subject}.svg", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # 2. Performance degradation plot (all subjects)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance Degradation by Pathway Noise", fontsize=14, fontweight="bold")

    for idx, (condition, title) in enumerate([
        ("convergent_noisy", "Convergent Pathway Noisy"),
        ("divergent_noisy", "Divergent Pathway Noisy"),
    ]):
        ax = axes[idx]

        for subject in subjects:
            subj_df = results_df[
                (results_df["subject"] == subject) & (results_df["condition"] == condition)
            ].sort_values("noise_std")

            # Calculate performance drop from baseline
            baseline_acc = subj_df[subj_df["noise_std"] == 0.0]["accuracy"].values[0]
            subj_df["acc_drop"] = baseline_acc - subj_df["accuracy"]

            ax.plot(
                subj_df["noise_std"],
                subj_df["acc_drop"],
                marker="o",
                label=subject,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Noise Standard Deviation", fontsize=12)
        ax.set_ylabel("Accuracy Drop from Baseline", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "performance_degradation_all_subjects.png", dpi=300, bbox_inches="tight")
        plt.savefig(save_dir / "performance_degradation_all_subjects.svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 3. Summary bar plot at highest noise level
    max_noise = results_df["noise_std"].max()
    high_noise_df = results_df[results_df["noise_std"] == max_noise]

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["convergent_noisy", "divergent_noisy"]
    x = np.arange(len(subjects))
    width = 0.35

    for idx, (condition, label, color) in enumerate([
        ("convergent_noisy", "Convergent noisy", "coral"),
        ("divergent_noisy", "Divergent noisy", "skyblue"),
    ]):
        accuracies = []
        for subject in subjects:
            acc = high_noise_df[
                (high_noise_df["subject"] == subject) & (high_noise_df["condition"] == condition)
            ]["accuracy"].values[0]
            accuracies.append(acc)

        ax.bar(x + idx * width, accuracies, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Performance Comparison at Highest Noise Level (σ={max_noise})", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "high_noise_comparison.png", dpi=300, bbox_inches="tight")
        plt.savefig(save_dir / "high_noise_comparison.svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 4. Average performance across all subjects with confidence intervals
    if len(subjects) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Average Performance Across All Subjects (with 95% CI)", fontsize=16, fontweight="bold")

        metrics = ["accuracy", "auroc", "f1", "sensitivity", "specificity"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            for condition, label, color, marker in [
                ("convergent_noisy", "Convergent pathway noisy", "coral", "o"),
                ("divergent_noisy", "Divergent pathway noisy", "skyblue", "s"),
            ]:
                # Get baseline values for each subject
                baseline_values = {}
                for subject in subjects:
                    baseline = results_df[
                        (results_df["subject"] == subject) &
                        (results_df["condition"] == condition) &
                        (results_df["noise_std"] == 0.0)
                    ][metric].values[0]
                    baseline_values[subject] = baseline

                # Calculate mean and CI for each noise level
                noise_levels = sorted(results_df["noise_std"].unique())
                means = []
                lower_bounds = []
                upper_bounds = []

                for noise in noise_levels:
                    drops = []
                    for subject in subjects:
                        baseline = baseline_values[subject]
                        current = results_df[
                            (results_df["subject"] == subject) &
                            (results_df["condition"] == condition) &
                            (results_df["noise_std"] == noise)
                        ][metric].values[0]
                        drop = baseline - current
                        drops.append(drop)

                    mean_drop = np.mean(drops)
                    std_drop = np.std(drops, ddof=1) if len(drops) > 1 else 0
                    # 95% CI using t-distribution
                    ci = stats.t.interval(0.95, len(drops)-1, loc=mean_drop, scale=std_drop/np.sqrt(len(drops)))

                    means.append(mean_drop)
                    lower_bounds.append(ci[0])
                    upper_bounds.append(ci[1])

                # Plot with confidence interval
                ax.plot(noise_levels, means, marker=marker, label=label, linewidth=2, markersize=8, color=color)
                ax.fill_between(noise_levels, lower_bounds, upper_bounds, alpha=0.2, color=color)

            ax.set_xlabel("Noise Standard Deviation", fontsize=12)
            ax.set_ylabel(f"{metric.capitalize()} Drop from Baseline", fontsize=12)
            ax.set_title(f"{metric.capitalize()}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

        # Remove empty subplot
        fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / "average_performance_all_subjects.png", dpi=300, bbox_inches="tight")
            plt.savefig(save_dir / "average_performance_all_subjects.svg", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # 5. Statistical summary table
    print("\n" + "="*80)
    print("PATHWAY IMPORTANCE SUMMARY")
    print("="*80)

    for subject in subjects:
        print(f"\n{subject}:")
        subj_df = results_df[results_df["subject"] == subject]

        baseline = subj_df[
            (subj_df["condition"] == "convergent_noisy") & (subj_df["noise_std"] == 0.0)
        ]
        baseline_acc = baseline["accuracy"].values[0]

        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        for condition, pathway_name in [
            ("convergent_noisy", "Convergent"),
            ("divergent_noisy", "Divergent"),
        ]:
            cond_df = subj_df[subj_df["condition"] == condition].sort_values("noise_std")
            max_noise_row = cond_df[cond_df["noise_std"] == max_noise].iloc[0]

            acc_at_max = max_noise_row["accuracy"]
            acc_drop = baseline_acc - acc_at_max
            pct_drop = (acc_drop / baseline_acc) * 100

            print(f"  {pathway_name} pathway:")
            print(f"    Accuracy at max noise (σ={max_noise}): {acc_at_max:.4f}")
            print(f"    Drop: {acc_drop:.4f} ({pct_drop:.1f}%)")

    print("\n" + "="*80)


def main(results_file, save_dir=None):
    """
    Main function to generate plots from results file.

    Args:
        results_file: path to CSV file with pathway analysis results
        save_dir: directory to save plots (None = display only)
    """
    results_df = pd.read_csv(results_file)

    print(f"Loaded results from: {results_file}")
    print(f"Subjects: {results_df['subject'].unique()}")
    print(f"Noise levels: {sorted(results_df['noise_std'].unique())}")

    plot_pathway_comparison(results_df, save_dir)

    if save_dir:
        print(f"\nPlots saved to: {save_dir}")
    else:
        print("\nPlots displayed.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize pathway importance analysis results.")
    parser.add_argument(
        "-r",
        "--results-file",
        type=str,
        required=True,
        help="Path to pathway analysis results CSV file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: display only)",
    )

    args = parser.parse_args()

    main(args.results_file, args.output_dir)
