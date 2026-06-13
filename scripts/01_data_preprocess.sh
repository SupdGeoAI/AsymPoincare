#!/usr/bin/env bash
set -euo pipefail

cd src/1.data_preprocess

python generate_seq_sym.py --edge_path ../../datasets/edge/edge_3 --save_path ../../datasets/sequence_sym/seq_3 --num_paths 80 --path_length 400
python generate_seq_sym.py --edge_path ../../datasets/edge/edge_10 --save_path ../../datasets/sequence_sym/seq_10 --num_paths 80 --path_length 400
python generate_seq_asym.py --edge_path ../../datasets/edge/edge_3 --save_path ../../datasets/sequence_asym/seq_3 --num_paths 80 --path_length 400
python generate_seq_asym.py --edge_path ../../datasets/edge/edge_10 --save_path ../../datasets/sequence_asym/seq_10 --num_paths 80 --path_length 400