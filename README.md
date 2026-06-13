# AsymPoincaré: Mapping the Geometry of Asymmetric Intercity Mobility

Welcome to the official repository for our work, **"Mapping the geometry of asymmetric intercity mobility: A Poincaré embedding approach accounting for urban hierarchy and proximity"**! This repository provides the codebase and processed data products needed to reproduce the main experiments, visualizations, and analytical results reported in the paper.

## 📌 Overview

We propose an asymmetry-oriented hyperbolic embedding framework for modeling asymmetric intercity mobility interactions. By leveraging the geometric properties of hyperbolic space and recent advances in network representation learning, the proposed framework jointly preserves the **hierarchy** and **proximity** dimensions of city systems. Meanwhile, it explicitly captures **directional source-target asymmetries** through a bipartite graph abstraction, in which each city is represented by both a source node and a target node. The framework provides a unified geometric representation for analyzing asymmetric intercity interactions, where the radial coordinate encodes hierarchical status, the angular coordinate captures similarity or spatial proximity, and the distance between source and destination embeddings reflects directional imbalance.

## 📂 Repository Structure

The original intercity mobility data used in this study were obtained from Baidu Maps. Due to commercial restrictions and data usage agreements, the raw mobile phone signaling and location-based service data cannot be publicly released. To facilitate reproducibility, this repository provides processed data products and scripts, including:

1. Precomputed random-walk sequences for symmetric and asymmetric models.
2. Trained embeddings for the main models used in the paper:
   - symmetric Euclidean embeddings;
   - symmetric Poincaré embeddings;
   - asymmetry-oriented Poincaré embeddings.
3. Code for generating random-walk sequences from users' own origin-destination mobility data.
4. Scripts for model training, evaluation, visualization, and analysis.

```text
.
├── datasets/
│   ├── sequence_sym/
│   │   ├── seq_3
│   │   └── seq_10
│   ├── sequence_asym/
│   │   ├── seq_3
│   │   └── seq_10
│   └── code2name.csv
│
├── logs/
│   ├── v_seq_3(10)_euc_sym/
│   │   ├── embedding
│   │   └── settings.txt
│   ├── v_seq_3(10)_pc_sym/
│   │   ├── embedding
│   │   └── settings.txt
│   └── v_seq_3(10)_pc_asym/
│       ├── embedding
│       └── settings.txt
│
├── scripts/
│   ├── 01_data_preprocess.sh
│   ├── 02_model_training.sh
│   ├── 03_model_evaluation.sh
│   ├── 04_model_visualization.sh
│   └── 05_model_analysis.sh
│
├── src/
├── results/
├── requirements.txt
└── README.md
```

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage Guide

The experimental workflow consists of five main steps.

### 1. Data Preprocessing

Generate random-walk sequences from intercity mobility networks:

```bash
bash scripts/01_data_preprocess.sh
```

This step prepares the sequence data required for model training. Both symmetric and asymmetric random-walk sequences can be generated depending on the model configuration.

### 2. Model Training

Train the embedding models using the generated random-walk sequences:

```bash
bash scripts/02_model_training.sh
```

This script trains the major models used in the paper, including:

- **Node2Vec / symmetric Euclidean embeddings**
- **symmetric Poincaré embeddings**
- **asymmetry-oriented Poincaré embeddings**

The trained embeddings and corresponding settings are saved under the `logs/` directory.

### 3. Model Evaluation

Evaluate the learned embeddings using reconstruction-based metrics and network geometry statistics:

```bash
bash scripts/03_model_evaluation.sh
```

This step includes:

- calculating top-N reconstruction rates;
- comparing reconstruction performance across different neighborhood sizes;
- estimating the δ-hyperbolicity of intercity mobility networks.

### 4. Model Visualization

Visualize the learned embeddings in Euclidean and Poincaré spaces:

```bash
bash scripts/04_model_visualization.sh
```

This step helps examine whether the learned embeddings preserve meaningful structures of city systems, including hierarchical positions, angular proximity patterns, and source-target differences.

### 5. Model Analysis

Analyze the relationships among hierarchy, proximity, and asymmetry:

```bash
bash scripts/05_model_analysis.sh
```

This step reproduces the main analytical results of the paper, including:

- hierarchy distributions derived from symmetric and asymmetric Poincaré embeddings;
- clustering patterns along the angular proximity dimension;
- the relationship between asymmetry and urban hierarchy;
- regional heterogeneity in asymmetry along the proximity dimension;
- differences between non-holiday and holiday mobility networks.

## 📊 Results

All outputs are saved in the `results/` directory by default.

## 🔗 Contact

For questions, issues, or contributions, please open an issue in this repository or contact the authors directly.