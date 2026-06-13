#!/usr/bin/env bash
set -euo pipefail

cd src/3.model_evaluation

python reconstruction_sym.py --space euclidean --version seq_3_euc_sym --network_path ../../datasets/edge/edge_3
python reconstruction_sym.py --space euclidean --version seq_10_euc_sym --network_path ../../datasets/edge/edge_10
python reconstruction_sym.py --space poincare --version seq_3_pc_sym --network_path ../../datasets/edge/edge_3
python reconstruction_sym.py --space poincare --version seq_10_pc_sym --network_path ../../datasets/edge/edge_10
python reconstruction_asym.py --version seq_3_pc_asym --network_path ../../datasets/edge/edge_3
python reconstruction_asym.py --version seq_10_pc_asym --network_path ../../datasets/edge/edge_10
python plot_reconstruction_curve.py
python hyperbolicity_stats.py --data_path ../../datasets/edge/edge_3 --save_path ../../results/stats/hyperbolicity/edge_3.json
python hyperbolicity_stats.py --data_path ../../datasets/edge/edge_10 --save_path ../../results/stats/hyperbolicity/edge_10.json
python hyperbolicity_curve.py