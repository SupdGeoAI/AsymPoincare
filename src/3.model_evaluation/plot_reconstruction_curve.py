import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_data(seq_key, data_dict):
    plt.figure(figsize=(8, 5.5))
    neighbors = list(range(10, 41))
    xticks = list(range(10, 41, 5))
    plt.xticks(xticks)

    custom_colors = {
        "Euclidean (Symmetry)": "#EF767A",
        "Poincaré (Symmetry)": "#456990",
        "Poincaré (Asymmetry)": "#48C0AA"
    }

    for label, file_path in data_dict.items():
        data = load_pickle_data(file_path)
        data = [d * 100 for d in data]
        color = custom_colors.get(label, None)
        plt.plot(neighbors, data, label=label, color=color, linewidth=1.8, alpha=0.9)

    plt.xlabel("Neighborhood Size")
    plt.ylabel("Reconstruction Rate (%)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))
    plt.grid(True, linewidth=0.6, color='gray', alpha=0.6)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"../../results/figures/reconstruction/Reconstruction_{seq_key}.png",
                dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    data_paths = {
        "March": {
            "Euclidean (Symmetry)": "../../results/stats/reconstruction/reconstruction_seq_3_euc_sym",
            "Poincaré (Symmetry)": "../../results/stats/reconstruction/reconstruction_seq_3_pc_sym",
            "Poincaré (Asymmetry)": "../../results/stats/reconstruction/reconstruction_seq_3_pc_asym"
        },
        "October": {
            "Euclidean (Symmetry)": "../../results/stats/reconstruction/reconstruction_seq_10_euc_sym",
            "Poincaré (Symmetry)": "../../results/stats/reconstruction/reconstruction_seq_10_pc_sym",
            "Poincaré (Asymmetry)": "../../results/stats/reconstruction/reconstruction_seq_10_pc_asym"
        }
    }

    for seq, paths in data_paths.items():
        plot_data(seq, paths)