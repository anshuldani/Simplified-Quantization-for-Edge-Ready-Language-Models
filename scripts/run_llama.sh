#!/bin/bash
# scripts/run_llama.sh
# Milestone C: Full LLaMA-3.2-1B pipeline on IIT A100 cluster
# Submit: sbatch scripts/run_llama.sh

#SBATCH --job-name=salient-quant-llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/llama_%j.out
#SBATCH --error=logs/llama_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adani@hawk.iit.edu

set -e

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================="

module load cuda/12.1
module load anaconda3

conda activate salient-quant

PROJECT_DIR=$HOME/salient-quant
cd $PROJECT_DIR

# Verify HF token for LLaMA access
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token"
    exit 1
fi

huggingface-cli login --token $HF_TOKEN

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

echo ""
echo "--- Full Milestone C: LLaMA-3.2-1B ---"
python experiments/run_experiment.py \
    --config configs/llama_full.yaml

echo ""
echo "========================================="
echo "Milestone C complete!"
echo "Results: results/llama_full/"
echo "End: $(date)"
echo "========================================="
