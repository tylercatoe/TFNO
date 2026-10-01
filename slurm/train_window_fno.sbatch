#!/bin/bash
#SBATCH --job-name=window_fno
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/window_fno_%j.out
#SBATCH --error=slurm/window_fno_%j.err

set -euo pipefail

cd /scratch/tcatoe/Turb_NO_26/TFNO

module purge
module load ngc/pytorch/23.06

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

window_data_dir="${1:-data/turpy_chunks_4km}"
window_output_dir="${2:-checkpoints/window_fno_4km_turbulent}"

python train_window_fno.py \
  --data-dir "$window_data_dir" \
  --output-dir "$window_output_dir" \
  --window 10 \
  --rollout-steps 10 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --weight-decay 1e-3 \
  --width 16 \
  --layers 3 \
  --modes-y 16 \
  --modes-x 16 \
  --loss blend \
  --scheduler plateau \
  --early-stopping-patience 20 \
  --blend-mse-weight 0.1 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --split-unit mode-combination \
  --grad-clip 1.0 \
  --seed 123 \
  --workers 0 \
  --device cuda
