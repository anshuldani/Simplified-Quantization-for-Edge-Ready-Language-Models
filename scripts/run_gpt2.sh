#!/bin/bash
# scripts/run_gpt2.sh
# Milestone B: Full GPT-2 pipeline on IIT A100 cluster
# Submit: sbatch scripts/run_gpt2.sh

#SBATCH --job-name=salient-quant-gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gpt2_%j.out
#SBATCH --error=logs/gpt2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adani@hawk.iit.edu

set -e

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================="

# Environment setup
module load cuda/12.1
module load anaconda3

conda activate salient-quant

PROJECT_DIR=$HOME/salient-quant
cd $PROJECT_DIR

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "--- PHASE 1: Quick sanity check (gpt2-small, 128 samples) ---"
python experiments/run_experiment.py \
    --config configs/gpt2_quick.yaml

echo ""
echo "--- PHASE 2: Full Milestone B (gpt2-medium, 512 samples, ablations) ---"
python experiments/run_experiment.py \
    --config configs/gpt2_full.yaml

echo ""
echo "--- PHASE 3: Bottleneck profiling ---"
python experiments/profile_model.py \
    --model gpt2-medium \
    --output results/gpt2_full/profiling

echo ""
echo "========================================="
echo "Milestone B complete!"
echo "Results: results/gpt2_full/"
echo "End: $(date)"
echo "========================================="
