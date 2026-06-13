import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

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

def get_od_midpoint_and_asymmetry(emb_dict):
    x_mid, y_mid, angle_deg, asymmetry, labels = [], [], [], [], []
    
    for k, v in emb_dict.items():
        kid = int(k)
        if kid >= 1000000:
            continue
        out_k = str(kid + 1000000)
        if out_k not in emb_dict:
            continue
        v_in = np.asarray(v, dtype=float)
        v_out = np.asarray(emb_dict[out_k], dtype=float)
        dist_in_out = PoincareDistance(v_in, v_out)
        if np.isnan(dist_in_out):
            continue
        x = (v_in[0] + v_out[0]) / 2.0
        y = (v_in[1] + v_out[1]) / 2.0
        x_mid.append(x)
        y_mid.append(y)
        theta = np.arctan2(y, x)
        deg = np.degrees(theta)
        if deg < 0:
            deg += 360.0
        angle_deg.append(deg)
        asymmetry.append(dist_in_out)
        labels.append(str(k))
    return (np.array(x_mid), np.array(y_mid), np.array(angle_deg), np.array(asymmetry), np.array(labels))

def circular_distance_deg(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, 360.0 - d)

def knn_weights_angular(angles_deg, k=8):
    angles_deg = np.asarray(angles_deg).flatten()
    n = len(angles_deg)
    W = np.zeros((n, n))
    for i in range(n):
        dist = circular_distance_deg(angles_deg[i], angles_deg)
        dist[i] = np.inf
        idx = np.argsort(dist)[:k]
        W[i, idx] = 1.0
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    W = W / row_sum
    return W

def moran_i(y, W):
    y = np.asarray(y).flatten()
    n = len(y)
    y_mean = y.mean()
    y_centered = y - y_mean
    s0 = W.sum()
    if s0 == 0:
        return np.nan, np.nan, np.nan, np.nan
    num = float(y_centered @ W @ y_centered)
    denom = (y_centered ** 2).sum()
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan
    I = (n / s0) * (num / denom)
    E_I = -1.0 / (n - 1) if n > 1 else 0
    if n <= 3:
        return I, E_I, np.nan, np.nan
    s1 = 0.5 * ((W + W.T) ** 2).sum()
    s2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    s3 = (y_centered ** 4).sum() / n / ((y_centered ** 2).sum() / n) ** 2
    s4 = (n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2
    s5 = (n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2
    var_I = ((n * s4 - s3 * s5) / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2)) - (E_I ** 2)
    if var_I <= 0:
        var_I = 1e-12
    z = (I - E_I) / np.sqrt(var_I)
    return I, E_I, var_I, z

def moran_permutation(y, W, n_perm=999, seed=42):
    I_obs, E_I, var_I, z = moran_i(y, W)
    if np.isnan(I_obs):
        return I_obs, np.nan, np.array([])
    n = len(y)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_perm, dtype=float)
    count = 0
    for i in range(n_perm):
        perm = rng.permutation(n)
        I_perm, _, _, _ = moran_i(y[perm], W)
        sims[i] = I_perm
        if not np.isnan(I_perm) and abs(I_perm) >= abs(I_obs):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return I_obs, p_value, sims

def get_candidate():
    return [
        '沈阳市', '长春市', '哈尔滨市', '乌鲁木齐市', '西宁市', '银川市', '兰州市', '西安市',
        '呼和浩特市', '北京市', '天津市', '石家庄市', '太原市', '南京市', '上海市', '合肥市',
        '杭州市', '福州市', '南昌市', '济南市', '郑州市', '长沙市', '武汉市', '广州市', '南宁市',
        '海口市', '拉萨市', '重庆市', '成都市', '贵阳市', '昆明市'
    ]

def get_candidate_map():
    return {
        '沈阳市': 'Shenyang',
        '长春市': 'Changchun',
        '哈尔滨市': 'Harbin',
        '乌鲁木齐市': 'Urumqi',
        '西宁市': 'Xining',
        '银川市': 'Yinchuan',
        '兰州市': 'Lanzhou',
        '西安市': 'Xi\'an',
        '呼和浩特市': 'Hohhot',
        '北京市': 'Beijing',
        '天津市': 'Tianjin',
        '石家庄市': 'Shijiazhuang',
        '太原市': 'Taiyuan',
        '南京市': 'Nanjing',
        '上海市': 'Shanghai',
        '合肥市': 'Hefei',
        '杭州市': 'Hangzhou',
        '福州市': 'Fuzhou',
        '南昌市': 'Nanchang',
        '济南市': 'Jinan',
        '郑州市': 'Zhengzhou',
        '长沙市': 'Changsha',
        '武汉市': 'Wuhan',
        '广州市': 'Guangzhou',
        '南宁市': 'Nanning',
        '海口市': 'Haikou',
        '拉萨市': 'Lhasa',
        '重庆市': 'Chongqing',
        '成都市': 'Chengdu',
        '贵阳市': 'Guiyang',
        '昆明市': 'Kunming'
    }

def convert(code_path):
    code2names = {}
    names2code = {}
    prov2idx = {}
    count = 1
    codes = pd.read_csv(code_path, delimiter=",")
    nums = codes.shape[0]
    for i in range(nums):
        code = str(codes['code'][i])
        name = str(codes['name'][i])
        prov = str(codes['prov'][i])
        if prov not in prov2idx:
            prov2idx[prov] = count
            count = count + 1
        code2names[code] = {'name': name, 'prov': prov}
        names2code[name] = code

    return code2names, names2code, prov2idx

def plot_moran_scatter(asymmetry, W, I_global, p_perm, version_label, labels):
    y = np.asarray(asymmetry).flatten()
    y_std = (y - y.mean()) / (y.std() or 1.0)
    lag = W @ y_std
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_std, lag, alpha=0.65, s=28, c='steelblue', edgecolors='none')
    slope = float((y_std @ lag) / (y_std @ y_std)) if np.any(y_std) else 0.0
    x_line = np.array([y_std.min(), y_std.max()])
    ax.plot(x_line, slope * x_line, 'r-', lw=2, label=f"slope = {slope:.3f}")
    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(0, color='gray', ls='--', alpha=0.7)
    
    code_path = '../../datasets/code2name.csv'
    code2names, names2code, prov2idx = convert(code_path)

    city_candidate = get_candidate()
    candidate_map = get_candidate_map()

    texts = []
    for idx, k in enumerate(labels):
        name = code2names[k]['name']
        if name in city_candidate:
            text = plt.text(y_std[idx], lag[idx], candidate_map[name], fontsize=6, zorder=4)
            texts.append(text)
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, expand_points=(1.2, 1.4), force_text=0.6, arrowprops=dict(arrowstyle='-', color='gray'))
    
    ax.text(0.97, 0.97, f"Moran's I = {I_global:.3f}\np = {p_perm:.4f}", transform=ax.transAxes, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='lightgray', alpha=0.9))
    ax.set_xlabel('Asymmetry (standardized)')
    ax.set_ylabel('Spatial Lag')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(f"../../results/figures/moran_scatter/moran_scatter_{version_label}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_moran(embed_path, version_label, k_neighbors, n_perm):
    emb_dict = load_embedding(embed_path)
    x_mid, y_mid, angle_deg, asymmetry, labels = get_od_midpoint_and_asymmetry(emb_dict)
    
    W = knn_weights_angular(angle_deg, k=k_neighbors)
    I_global, p_perm, I_perm_sims = moran_permutation(asymmetry, W, n_perm=n_perm)
    _, E_I, var_I, z = moran_i(asymmetry, W)
    print(f"[{version_label}] n={len(asymmetry)}, Moran's I = {I_global:.4f}, p (perm) = {p_perm:.4f}, E[I] = {E_I:.4f}, z = {z:.4f}")
    plot_moran_scatter(asymmetry, W, I_global, p_perm, version_label, labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--n_perm', type=int, default=999)
    args = parser.parse_args()

    args.embed_path = f"../../logs/v_{args.version}/embedding"
    run_moran(args.embed_path, args.version, args.k, args.n_perm)

if __name__ == '__main__':
    main()