import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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
    for k, v in emb_dict.items():
        v = np.asarray(v, dtype=float)
        d = PoincareDistance(v, origin)
        if not np.isnan(d) and d > 0:
            hierarchy.append(d)
    return np.array(hierarchy)

def get_od_hierarchies(emb_dict):
    origin = np.array([0.0, 0.0])
    h = []

    for k, v in emb_dict.items():
        kid = int(k)
        if kid >= 1000000:
            continue
        out_k = str(kid + 1000000)
        if out_k not in emb_dict:
            continue
        v_in = np.asarray(v, dtype=float)
        v_out = np.asarray(emb_dict[out_k], dtype=float)
        d_in = PoincareDistance(v_in, origin)
        d_out = PoincareDistance(v_out, origin)
        if not (np.isnan(d_in) or np.isnan(d_out)) and d_in > 0 and d_out > 0:
            h.append((d_in + d_out)/2.0)

    return np.array(h)

def build_rank_size_df(values):
    values = np.asarray(values).flatten()
    values = values[values > 0]
    if len(values) == 0:
        return pd.DataFrame()
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    rank = np.arange(1, len(sorted_vals) + 1, dtype=float)
    df = pd.DataFrame({
        'rank': rank,
        'log_rank': np.log10(rank),
        'log_size': -sorted_vals
    })
    return df

def fit_power_law(df, x_col='log_rank', y_col='log_size'):
    x = df[x_col].values
    y = df[y_col].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return slope, intercept, r_value, r_squared, p_value

def plot_rank_size(ax, df, label, show_fit=True, text_anchor='top_left'):
    if df is None or len(df) == 0:
        return
    ax.scatter(df['log_rank'], df['log_size'], alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    if show_fit:
        slope, intercept, r_value, r_squared, p_value = fit_power_law(df)
        
        x_fit = np.linspace(df['log_rank'].min(), df['log_rank'].max(), 100)
        y_fit = slope * x_fit + intercept

        ax.plot(
            x_fit,
            y_fit,
            '--',
            linewidth=2,
            label=f'Seg (R²={r_squared:.2f}, s={slope:.2f})'
        )

def plot_fig(df, label, version):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_rank_size(ax, df, label)
    ax.set_xlabel('log10(Rank)', fontsize=16)
    ax.set_ylabel('-Hierarchy', fontsize=16)
    ax.legend(loc='best', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = os.path.join('../../results/figures/rank_size/', f'{version}_rank_size_{label}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str)
    parser.add_argument('--asymmetric', action='store_true')
    args = parser.parse_args()

    embed_path = f"../../logs/v_{args.version}/embedding"
    emb_dict = load_embedding(embed_path)

    if not args.asymmetric:
        hierarchy = get_hierarchies(emb_dict)
        print(f"Nodes: {len(hierarchy)}")
        print(f"Hierarchy: [{hierarchy.min():.4f}, {hierarchy.max():.4f}]")
        df = build_rank_size_df(hierarchy)
        plot_fig(df,"Symmetric",args.version)
    else:
        hierarchy = get_od_hierarchies(emb_dict)
        print(f"Nodes: {len(hierarchy)}")
        print(f"Hierarchy: [{hierarchy.min():.4f}, {hierarchy.max():.4f}]")
        df = build_rank_size_df(hierarchy)
        plot_fig(df,"Asymmetric",args.version)

if __name__ == '__main__':
    main()