import math
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed


OUTPUT_DIR = './result_point_patterns_od'
PANEL_PATH = os.path.join(OUTPUT_DIR, 'point_patterns_od_panel.svg')


def angle_with_x_axis(x, y):
    theta_rad = math.atan2(y, x)
    theta_deg = math.degrees(theta_rad)
    return theta_deg if theta_deg >= 0 else theta_deg + 360


def PoincareDistance(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u >= 1 or norm_v >= 1:
        return np.nan
    euclidean_dist = np.linalg.norm(u - v)
    return float(np.arccosh(1 + 2 * (euclidean_dist ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2))))


def load_embedding(embed_path):
    with open(embed_path, 'rb') as f:
        return pickle.load(f)


def getSymData(emb_dict):
    degrees = []
    hierarchies = []
    for _, v in emb_dict.items():
        v = np.asarray(v, dtype=float)
        degree = angle_with_x_axis(v[0], v[1])
        hierarchy = PoincareDistance(v, np.array([0.0, 0.0]))
        if not np.isnan(hierarchy):
            degrees.append(degree)
            hierarchies.append(hierarchy)
    return np.array(degrees), np.array(hierarchies)


def getAsymData(emb_dict):
    degrees_origin = []
    degrees_destination = []
    hierarchies_origin = []
    hierarchies_destination = []

    for k, v in emb_dict.items():
        kid = int(k)
        if kid >= 1000000:
            continue
        out_k = str(kid + 1000000)
        if out_k not in emb_dict:
            continue

        v_in = np.asarray(v, dtype=float)
        v_out = np.asarray(emb_dict[out_k], dtype=float)
        degree_in = angle_with_x_axis(v_in[0], v_in[1])
        degree_out = angle_with_x_axis(v_out[0], v_out[1])
        hierarchy_in = PoincareDistance(v_in, np.array([0.0, 0.0]))
        hierarchy_out = PoincareDistance(v_out, np.array([0.0, 0.0]))

        if not np.isnan(hierarchy_in) and not np.isnan(hierarchy_out):
            degrees_origin.append(degree_in)
            degrees_destination.append(degree_out)
            hierarchies_origin.append(hierarchy_in)
            hierarchies_destination.append(hierarchy_out)

    return (
        np.array(degrees_origin),
        np.array(hierarchies_origin),
        np.array(degrees_destination),
        np.array(hierarchies_destination),
    )


def circular_distance_matrix(angles):
    angles = np.array(angles)
    diff_matrix = np.abs(angles[:, None] - angles[None, :])
    return np.minimum(diff_matrix, 360 - diff_matrix)


def circular_K_function_fast(angles, d_values):
    angles = np.array(angles)
    N = len(angles)
    lambd = N / 360

    dist_matrix = circular_distance_matrix(angles)
    iu = np.triu_indices(N, k=1)
    pairwise_dists = dist_matrix[iu]

    K_vals = []
    for d in d_values:
        count = np.sum(pairwise_dists <= d)
        K_vals.append(2 * count / (lambd * N))
    return np.array(K_vals)


def simulate_CSR_K_single_run(N, d_values, seed):
    rng = np.random.default_rng(seed)
    sim_angles = rng.uniform(0, 360, size=N)
    return circular_K_function_fast(sim_angles, d_values) - 2 * d_values


def compute_l_function(angles, n_simulations, seed, n_jobs):
    d_values = np.linspace(1, 90, 90)
    k_vals = circular_K_function_fast(angles, d_values)
    l_vals = k_vals - 2 * d_values

    simulations = Parallel(n_jobs=n_jobs)(
        delayed(simulate_CSR_K_single_run)(len(angles), d_values, seed + i)
        for i in tqdm(range(n_simulations), desc='Simulating CSR')
    )
    simulations = np.array(simulations)
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)
    return d_values, l_vals, lower, upper


def plot_l_function(ax, angles, n_simulations, seed, n_jobs):
    d_values, l_vals, lower, upper = compute_l_function(angles, n_simulations, seed, n_jobs)

    ax.plot(d_values, l_vals, linewidth=2.2, alpha=0.95)
    ax.fill_between(d_values, lower, upper, color='gray', alpha=0.22)
    ax.axhline(0, color='black', linestyle=':', linewidth=1.8)
    ax.set_xlim(0, 95)
    ax.set_xticks(np.arange(0, 91, 10))
    ax.set_xlabel('Degrees', fontsize=10)
    ax.set_ylabel('L-functions', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulations', type=int, default=999)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sym_march, _ = getSymData(load_embedding('../../logs/v_seq_3_pc_sym/embedding'))
    sym_october, _ = getSymData(load_embedding('../../logs/v_seq_10_pc_sym/embedding'))
    asym_march_origin, _, asym_march_destination, _ = getAsymData(
        load_embedding('../../logs/v_seq_3_pc_asym/embedding')
    )
    asym_october_origin, _, asym_october_destination, _ = getAsymData(
        load_embedding('../../logs/v_seq_10_pc_asym/embedding')
    )

    panels = [
        [sym_march, sym_october],
        [asym_march_origin, asym_october_origin],
        [asym_march_destination, asym_october_destination],
    ]
    row_labels = [
        'Symmetric Embedding',
        'Asymmetric Embedding\n(Origin)',
        'Asymmetric Embedding\n(Destination)',
    ]
    col_titles = ['March', 'October']

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 11.3))

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=18, fontweight='bold', pad=10)

    seed_offset = 0
    for row in range(3):
        for col in range(2):
            plot_l_function(axes[row, col], panels[row][col], args.simulations, args.seed + seed_offset, args.n_jobs)
            seed_offset += args.simulations + 17

    for row, label in enumerate(row_labels):
        fig.text(
            0.025,
            0.83 - row * 0.315,
            label,
            rotation=90,
            va='center',
            ha='center',
            fontsize=16,
            fontweight='bold',
        )

    plt.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.06, hspace=0.26, wspace=0.18)
    plt.savefig(PANEL_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(PANEL_PATH)


if __name__ == '__main__':
    main()
