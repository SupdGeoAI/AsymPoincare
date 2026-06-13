#!/usr/bin/env bash
set -euo pipefail

cd src/2.model_training

python train_euclidean.py --version seq_3_euc_sym --data_path ../../datasets/sequence_sym/seq_3 --lr 0.01
python train_euclidean.py --version seq_10_euc_sym --data_path ../../datasets/sequence_sym/seq_10 --lr 0.01
python train_poincare_sym.py --version seq_3_pc_sym --data_path ../../datasets/sequence_sym/seq_3 --lr 0.001
python train_poincare_sym.py --version seq_10_pc_sym --data_path ../../datasets/sequence_sym/seq_10 --lr 0.001
python train_poincare_asym.py --version seq_3_pc_asym --data_path ../../datasets/sequence_asym/seq_3 --lr 0.001 --r 0.9
python train_poincare_asym.py --version seq_10_pc_asym --data_path ../../datasets/sequence_asym/seq_10 --lr 0.001 --r 0.9