#!/usr/bin/env bash
set -euo pipefail

cd src/5.model_analysis

python rank_size_plot.py --version seq_3_pc_sym
python rank_size_plot.py --version seq_10_pc_sym
python rank_size_plot.py --version seq_3_pc_asym --asymmetric
python rank_size_plot.py --version seq_10_pc_asym --asymmetric
python point_patterns_plot.py --version seq_3_pc_sym
python point_patterns_plot.py --version seq_10_pc_sym
python point_patterns_plot.py --version seq_3_pc_asym --asymmetric
python point_patterns_plot.py --version seq_10_pc_asym --asymmetric
python asymmetry_to_hierarchy.py
python asymmetry_to_proximity.py --version seq_3_pc_asym
python asymmetry_to_proximity.py --version seq_10_pc_asym
python moran_asymmetry_proximity.py --version seq_3_pc_asym
python moran_asymmetry_proximity.py --version seq_10_pc_asym