#!/bin/bash
# SBATCH -n 4 # Number of cores, pero no funciona
# SBATCH -N 1 # Ensure that all cores are on one machine
# SBATCH -D /home/smasip/comicsv1 # working directory
# SBATCH -t 3-00:00 # Runtime in D-HH:MM
# SBATCH -p dag # Partition to submit to
# SBATCH --mem 24000 # MB solicitados
# SBATCH --gres gpu:1080Ti:2 # GPU's solicitadas gpu: o gpu:TitanXp: o gpu:1080Ti:
# SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
# SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written
python /home/smasip/comicsv1/hf_trainer.py
#free -h