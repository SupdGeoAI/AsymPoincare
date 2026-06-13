import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


OUTPUT_DIR = './result_rank_size_od'
PANEL_PATH = os.path.join(OUTPUT_DIR, 'rank_size_od_panel.svg')


def load_embedding(embed_path):
    with open(embed_path, 'rb') as f:
        return pickle.load(f)


def PoincareDistance(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u >= 1 or norm_v >= 1:
        return np.nan
    euclidean_dist = np.linalg.norm(u - v)
    return float(np.arccosh(1 + 2 * (euclidean_dist ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2))))


def get_hierarchies(emb_dict):
    origin = np.array([0.0, 0.0])
    hierarchy = []
    for _, v in emb_dict.items():
        d = PoincareDistance(v, origin)
        if not np.isnan(d) and d > 0:
            hierarchy.append(d)
    return np.array(hierarchy)


def get_od_hierarchies(emb_dict):
    origin = np.array([0.0, 0.0])
    h_origin = []
    h_destination = []

    for k, v in emb_dict.items():
        kid = int(k)
        if kid >= 1000000:
            continue
        out_k = str(kid + 1000000)
        if out_k not in emb_dict:
            continue
        d_in = PoincareDistance(v, origin)
        d_out = PoincareDistance(emb_dict[out_k], origin)
        if not (np.isnan(d_in) or np.isnan(d_out)) and d_in > 0 and d_out > 0:
            h_origin.append(d_in)
            h_destination.append(d_out)

    return np.array(h_origin), np.array(h_destination)


def build_rank_size_df(values):
    values = np.asarray(values).flatten()
    values = values[values > 0]
    sorted_vals = np.sort(values)
    rank = np.arange(1, len(sorted_vals) + 1, dtype=float)
    return pd.DataFrame({
        'log_rank': np.log10(rank),
        'log_size': -sorted_vals,
    })


def fit_power_law(df):
    x = df['log_size'].values
    y = df['log_rank'].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value ** 2


def plot_rank_size(ax, values):
    df = build_rank_size_df(values)
    slope, intercept, r_squared = fit_power_law(df)

    ax.scatter(
        df['log_size'],
        df['log_rank'],
        alpha=0.65,
        s=18,
        edgecolors='black',
        linewidths=0.45,
    )

    x_fit = np.linspace(df['log_size'].min(), df['log_size'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(
        x_fit,
        y_fit,
        '--',
        linewidth=2.4,
        label='Linear fit',
    )
    ax.plot([], [], linestyle='None', label=f"R² = {r_squared:.2f}")
    ax.plot([], [], linestyle='None', label=f"Slope = {slope:.2f}")
    ax.plot([], [], linestyle='None', label=f"Intercept = {intercept:.2f}")

    ax.set_xlabel('-Hierarchy', fontsize=10)
    ax.set_ylabel('log10(Rank)', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9, handlelength=2.2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sym_march = get_hierarchies(load_embedding('../../logs/v_seq_3_pc_sym/embedding'))
    sym_october = get_hierarchies(load_embedding('../../logs/v_seq_10_pc_sym/embedding'))
    asym_march_origin, asym_march_destination = get_od_hierarchies(
        load_embedding('../../logs/v_seq_3_pc_asym/embedding')
    )
    asym_october_origin, asym_october_destination = get_od_hierarchies(
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

    fig, axes = plt.subplots(3, 2, figsize=(10.6, 11.3))

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=18, fontweight='bold', pad=10)

    for row in range(3):
        for col in range(2):
            plot_rank_size(axes[row, col], panels[row][col])

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

    plt.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.06, hspace=0.26, wspace=0.13)
    plt.savefig(PANEL_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(PANEL_PATH)


if __name__ == '__main__':
    main()
