#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=COMPUTE-ACCOUNT-HERE
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=YOUR-EMAIL-HERE
#SBATCH --mail-type=ALL

module load python/3.10.13
module load cuda/12.2
module load arrow/17.0.0
module load rust/1.70.0
module load gcc/12.3
module load python/3.10.13
 
echo "Python version: "
python --version

# export WANDB_API_KEY=WANDB-API-KEY-HERE
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_MODE=offline

export HF_HOME=/scratch/hsila/hfcache
export HF_DATASETS_CACHE=/scratch/hsila/hfcache/datasets
# export HF_TOKEN=HF-TOKEN-HERE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

echo -e "\n\nInstalling pymongo"
pip install --no-index ./wheels/pymongo/*.whl
ec=$?
echo -e "pymongo done with exit code $ec\n\n"

echo -e "\n\nInstalling s3fs"
pip install --no-index ./wheels/s3fs/*.whl
ec=$?
echo -e "s3fs done with exit code $ec\n\n"

echo -e "\n\nInstalling requirements"
pip install --no-index -r requirements.txt
ec=$?
echo -e "Requirements done with exit code $ec\n\n"

python -c "import torch; print(f'torch cuda available: {torch.cuda.is_available()}')"
pip install --no-index psutil==5.9.8

echo -e "\n\nInstalling deepspeed"
pip install --no-index ./wheels/deepspeed/*.whl
ec=$?
echo -e "deepspeed done with exit code $ec\n\n"

echo -e "\n\nInstalling flash-att"
pip install --no-index ./wheels/flash-attention/*.whl
ec=$?
echo -e "flash-att done with exit code $ec\n\n"

echo -e "\n\nInstalling contrastors"
pip install --no-index -e .
ec=$?
echo -e "Contrastors done with exit code $ec\n\n"

cd PATH-TO-THIS-REPO

pip install --no-index wheel packaging ninja setuptools
python -c "import torch; print(f'torch cuda available: {torch.cuda.is_available()}')"

torchrun --nproc-per-node=4 train.py --config=configs/train/chem_contrastive_finetune.yaml --dtype=bf16

