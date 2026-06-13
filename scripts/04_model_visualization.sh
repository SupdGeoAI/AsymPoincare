#!/usr/bin/env bash
set -euo pipefail

cd src/4.model_visualization

python visualize_euclidean.py --version seq_3_euc_sym
python visualize_euclidean.py --version seq_10_euc_sym
python visualize_poincare_sym.py --version seq_3_pc_sym
python visualize_poincare_sym.py --version seq_10_pc_sym
python visualize_poincare_asym.py --version seq_3_pc_asym
python visualize_poincare_asym.py --version seq_10_pc_asym