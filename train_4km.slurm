#!/bin/bash
#SBATCH --job-name=fno_4km
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fno_4km_%j.out
#SBATCH --error=logs/fno_4km_%j.err

set -euo pipefail

cd /scratch/tcatoe/Turb_NO_26/TFNO

module purge
module load ngc/pytorch/23.06

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if ! compgen -G "turpy_chunks_4km/chunk_*.pt" > /dev/null; then
    echo "No TurPy chunks found in turpy_chunks_4km" >&2
    exit 1
fi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

python train.py \
  --data-dir turpy_chunks_4km \
  --chunk-pattern 'chunk_*.pt' \
  --output-dir checkpoints/turpy_fno_4km_ic_split \
  --split-unit mode-combination \
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
  --grad-clip 1.0 \
  --seed 123 \
  --workers 0 \
  --device cuda
