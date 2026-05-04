#!/bin/bash
#SBATCH --job-name=DMET_SQD_tpp_CO2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/%x_%j.out
#SBATCH --error=/scratch/%u/%x_%j.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dmet-sqd310

# Create job-specific scratch directory
SCRATCH_DIR=/scratch/$USER/tangelo_dmet_sqd_$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Limit BLAS threads (better performance for PySCF)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo "Running on $(hostname)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Run Python with unbuffered output
python -u ~/Tangelo_2.0_hardware/tpp_CO2_relaxed_scan.py

