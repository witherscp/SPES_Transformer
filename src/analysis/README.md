#  Run Noise analysis


 single subject:
```bash
python src/analysis/test_pathway_importance.py \
    --model-path experiments/Epat31_model_Fusion_seed_1_final_best_model.pt \
    --test-subjects Epat31
```

multiple subjects:
```bash
python src/analysis/test_pathway_importance.py \
    --model-path experiments/best_model.pt \
    --test-subjects Epat31 Spat37 Epat26
```

Custom noise levels:
```bash
python src/analysis/test_pathway_importance.py \
    --model-path experiments/best_model.pt \
    --test-subjects Epat31 \
    --noise-levels 0.0 0.05 0.1 0.25 0.5 1.0
```

###  Results Plots

Generate plots from results:
```bash
python src/analysis/plot_pathway_results.py \
    --results-file results/pathway_analysis/pathway_importance_20241205_123456.csv \
    --output-dir results/pathway_analysis/plots
```

Display plots without saving:
```bash
python src/analysis/plot_pathway_results.py \
    --results-file results/pathway_analysis/pathway_importance_20241205_123456.csv
```

## 

### Results CSV
Location: `results/pathway_analysis/pathway_importance_<timestamp>.csv`



# Using Accre

Running
```bash
source source venv/bin/activate
sbatch  sbatch run_pathway_analysis_gpu.sh

```

### Slurm script
```bash

#!/bin/bash
#SBATCH --account=p_dsi_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_titan_x:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=pathway_analysis_all
#SBATCH --output=logs/pathway_analysis_%j.out
#SBATCH --error=logs/pathway_analysis_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Load modules if needed (uncomment if required)
# module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Verify GPU is available
nvidia-smi
echo ""

# Run pathway analysis for all subjects
# The script will auto-detect the correct model for each subject
echo "Running pathway importance analysis for all subjects..."
python src/analysis/test_pathway_importance.py \
    --model-path auto \
    --test-subjects Epat26 Epat30 Epat31 Epat34 Epat35 Epat37 Epat38 Epat39 Spat30 Spat31 Spat34 Spat36 Spat37 Spat41 Spat42 Spat48 Spat49 Spat50 Spat52 Spat53 \
    --noise-levels 0.0 2.0 4.0 6.0 8.0 10.0

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo "Results saved to: results/pathway_analysis/"
else
    echo ""
    echo "Analysis failed! Check error log."
    exit 1
fi

echo ""
echo "End time: $(date)"

```


Plotting
```bash
python src/analysis/plot_pathway_results.py   --results-file /home/bibroce1/projects/results/pathway_analysis/pathway_importance_20251205_164827.csv   --output-dir results/pathway_analysis/plots

```
