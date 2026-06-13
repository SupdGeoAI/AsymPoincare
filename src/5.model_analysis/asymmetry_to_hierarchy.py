import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from adjustText import adjust_text
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, RANSACRegressor

def convert(code_path):
    code2names = {}
    names2code = {}
    prov2idx = {}
    count = 1
    codes = pd.read_csv(code_path,delimiter=",")
    nums = codes.shape[0]
    for i in range(nums):
        code = str(codes['code'][i])
        code_out = str(codes['code'][i]+1000000)
        name = str(codes['name'][i])
        prov = str(codes['prov'][i])
        if prov not in prov2idx:
            prov2idx[prov] = count
            count = count + 1
        code2names[code] = {'name':name,'prov':prov}
        code2names[code_out]= {'name':name,'prov':prov}
        names2code[name] = [code, code_out]

    return code2names, names2code, prov2idx

def PoincareDistance(u,v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u-v)
    return float(np.arccosh(1+2*(euclidean_dist**2)/((1-norm_u**2)*(1-norm_v**2))))

def getData(emb_dict,code2names):
    hierarchies = []
    asymmetries = []
    labels = []
    for k,v in emb_dict.items():
        if int(k) - 1000000 < 0:
            out_k = str(int(k) + 1000000)
            out_v = emb_dict[out_k]
            hierarchy_in = PoincareDistance(v,np.array([0.0,0.0]))
            hierarchy_out = PoincareDistance(out_v,np.array([0.0,0.0]))
            avg_hierarchy = (hierarchy_in + hierarchy_out)/2.0
            asymmetry = PoincareDistance(v,out_v)
            hierarchies.append(avg_hierarchy)
            asymmetries.append(asymmetry)
            labels.append(str(k))

    return hierarchies,asymmetries,labels

def compute_local_slope_by_x_distance(x, y, bandwidth=0.3):
    x = np.array(x)
    log_y = np.log(np.array(y))
    slopes = []

    for i in range(len(x)):
        center = x[i]
        mask = (x >= center - bandwidth) & (x <= center + bandwidth)
        if np.sum(mask) < 3:
            slopes.append(0)
            continue

        x_win = x[mask].reshape(-1, 1)
        y_win = log_y[mask]
        model = LinearRegression().fit(x_win, y_win)
        slopes.append(model.coef_[0])

    return np.array(slopes)

def fit_line_with_ransac(x, log_y, sample_weight=None, residual_threshold=0.9, min_samples=10, random_state=0):
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=residual_threshold,
        min_samples=min_samples,
        random_state=random_state
    )
    ransac.fit(x.reshape(-1, 1), log_y, sample_weight=sample_weight)
    model = ransac.estimator_
    inliers = ransac.inlier_mask_
    r2 = r2_score(log_y[inliers], ransac.predict(x[inliers].reshape(-1, 1)))
    x_fit = np.linspace(x[inliers].min(), x[inliers].max(), 100).reshape(-1, 1)
    y_fit = np.exp(model.predict(x_fit))
    return model, inliers, r2, x_fit.flatten(), y_fit

def get_city_dict():
    return {
        '黄石市': 'Huangshi',
        '武汉市': 'Wuhan',
        '孝感市': 'Xiaogan',
        '宁波市': 'Ningbo',
        '杭州市': 'Hangzhou',
        '上海市': 'Shanghai',
        '无锡市': 'Wuxi',
        '苏州市': 'Suzhou',
        '成都市': 'Chengdu',
        '重庆市': 'Chongqing',
        '北京市': 'Beijing',
        '天津市': 'Tianjin',
        '保定市': 'Baoding',
        '廊坊市': 'Langfang',
        '石家庄市': 'Shijiazhuang',
        '哈尔滨市': 'Harbin',
        '沈阳市': 'Shenyang',
        '西安市': 'Xi\'an',
        '洛阳市': 'Luoyang',
        '信阳市': 'Xinyang',
        '昆明市': 'Kunming',
        '泉州市': 'Quanzhou',
        '厦门市': 'Xiamen',
        '南昌市': 'Nanchang'
    }

def fit_and_plot_two_dominant_lines(
    x,
    y,
    version,
    bandwidth=0.6,
    residual_threshold=0.9,
    min_inliers_ratio=0.5,
    random_state=0,
    code2names=None,
    labels=None
    ):

    x = np.array(x)
    y = np.array(y)

    log_y = np.log(y)

    slopes = compute_local_slope_by_x_distance(x, y, bandwidth=bandwidth).reshape(-1, 1)
    cluster_labels = KMeans(n_clusters=2, random_state=random_state).fit(slopes).labels_

    point_colors = np.full(len(x), 'none', dtype=object)
    color_map = ['#1f77b4', 'orangered']
    color_map_line = ['crimson', 'orangered']
    lines = []

    main_label = 1
    sample_weight_main = (cluster_labels == main_label).astype(float)
    min_samples_main = max(int(min_inliers_ratio * np.sum(sample_weight_main)), 10)
    model_main, inliers_main, r2_main, x_fit_main, y_fit_main = fit_line_with_ransac(x, log_y, sample_weight=sample_weight_main, residual_threshold=residual_threshold, min_samples=min_samples_main, random_state=random_state)
    point_colors[inliers_main] = color_map[0]
    lines.append({
        'slope': model_main.coef_[0],
        'intercept': model_main.intercept_,
        'r2': r2_main,
        'x_fit': x_fit_main,
        'y_fit': y_fit_main,
        'inlier_mask': inliers_main
    })

    remain_mask = ~inliers_main
    x_remain, y_remain = x[remain_mask], y[remain_mask]
    log_y_remain = np.log(y_remain)
    log_y_pred_main = model_main.predict(x_remain.reshape(-1, 1))
    above_mask = log_y_remain > log_y_pred_main

    x_above = x_remain[above_mask]
    y_above = y_remain[above_mask]
    log_y_above = np.log(y_above)
    min_samples_secondary = max(int(min_inliers_ratio * len(x_above)), 2)

    model_sec, inliers_sec, r2_sec, x_fit_sec, y_fit_sec = fit_line_with_ransac( x_above, log_y_above, residual_threshold=residual_threshold, min_samples=min_samples_secondary, random_state=random_state)

    idx_remain = np.where(remain_mask)[0]
    idx_above = idx_remain[above_mask]
    idx_sec = idx_above[inliers_sec]
    point_colors[idx_sec] = color_map[1]
    lines.append({
        'slope': model_sec.coef_[0],
        'intercept': model_sec.intercept_,
        'r2': r2_sec,
        'x_fit': x_fit_sec,
        'y_fit': y_fit_sec,
        'inlier_mask': idx_sec
    })

    plt.figure(figsize=(10, 7))
    
    for color, idx in zip(color_map, [inliers_main, idx_sec]):
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            plt.scatter(x[idx], y[idx], c=color, edgecolors='k', alpha=0.7)
        else:
            plt.scatter(x[idx], y[idx], c=color, edgecolors='k', alpha=0.7)

    for i, line in enumerate(lines):
        plt.plot(line['x_fit'], line['y_fit'], color=color_map_line[i], linewidth=3,
                 label=f"Fit: $log(y) = {line['slope']:.2f}x + {line['intercept']:.2f}$ (R² = {line['r2']:.2f})")

    city_dict = get_city_dict()
    if code2names is not None and labels is not None:
        texts = []
        for idx in idx_sec:
            name = code2names[labels[idx]]['name']
            if name in city_dict:
                texts.append(plt.text(x[idx], y[idx], city_dict[name], fontsize=10))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), expand_text=(1.05, 1.2), expand_points=(1.05, 1.2), force_text=0.2, force_points=0.2, only_move={'points':'y', 'text':'xy'}, lim=100)

    plt.yscale('log')
    plt.xlabel("Hierarchies", fontsize=16)
    plt.ylabel("Asymmetries (log scale)", fontsize=16)
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../results/figures/asymmetry_to_hierarchy/To_hierarchies_' + version + '.png', dpi=300)

def plot_scatter_hierarchies_asymmetries(hierarchies, asymmetries, version, code2names, labels):
    x = np.array(hierarchies).reshape(-1, 1)
    y = np.array(asymmetries)
    log_y = np.log(y)
    model = LinearRegression()
    model.fit(x, log_y)
    a = model.coef_[0]
    b = model.intercept_
    log_y_pred = model.predict(x)
    y_pred = np.exp(log_y_pred)

    r2 = r2_score(log_y, log_y_pred)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.6, edgecolors='k')
    x_fit = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_fit = np.exp(model.predict(x_fit))
    plt.plot(x_fit, y_fit, color='crimson', lw=2.5, label=f'Fit: $log(y) = {a:.2f}x + {b:.2f}$ (R² = {r2:.2f})')
    plt.yscale('log')

    city_dict = get_city_dict()
    texts = []
    for idx, k in enumerate(labels):
        name = code2names[k]['name']
        if name in city_dict:
            texts.append(plt.text(x[idx], y[idx], city_dict[name], fontsize=10))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_text=(1.2, 1.4),
        expand_points=(1.5, 2.0),
        force_text=0.3,
        force_points=0.6,
        force_pull=0.02,
        only_move={'points': 'xy', 'text': 'xy'},
        ensure_inside_axes=True,
        lim=200
    )
    
    plt.xlabel("Hierarchies", fontsize=16)
    plt.ylabel("Asymmetries (log scale)", fontsize=16)
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    plt.savefig('../../results/figures/asymmetry_to_hierarchy/To_hierarchies_' + version + '.png', dpi=300)

def main(data_path_march, data_path_october):
    code_path = '../../datasets/code2name.csv'
    code2names, names2code, prov2idx = convert(code_path)

    with open(data_path_march,'rb') as f:
        emb_dict_march = pickle.load(f)
    hierarchies_march, asymmetries_march, labels_march = getData(emb_dict_march, code2names)
    fit_and_plot_two_dominant_lines(hierarchies_march, asymmetries_march, 'seq_3_pc_asym', code2names=code2names, labels=labels_march)

    with open(data_path_october,'rb') as f:
        emb_dict_october = pickle.load(f)
    hierarchies_october, asymmetries_october, labels_october = getData(emb_dict_october, code2names)
    plot_scatter_hierarchies_asymmetries(hierarchies_october, asymmetries_october, 'seq_10_pc_asym', code2names=code2names, labels=labels_october)

if __name__ == '__main__':
    data_path_march = f"../../logs/v_seq_3_pc_asym/embedding"
    data_path_october = f"../../logs/v_seq_10_pc_asym/embedding"
    
    main(data_path_march, data_path_october)