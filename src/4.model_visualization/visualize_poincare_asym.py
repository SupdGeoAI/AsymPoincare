import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import matplotlib.colors as colors
from adjustText import adjust_text

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

    return code2names,names2code,prov2idx

def EuclideanDistance(u,v):
    return float(np.linalg.norm(u-v))

def PoincareDistance(u,v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u-v)
    return float(np.arccosh(1+2*(euclidean_dist**2)/((1-norm_u**2)*(1-norm_v**2))))

def getData_size(emb_dict,code2names):
    x_vals = []
    y_vals = []
    labels = []
    asymmetries = []
    for k,v in emb_dict.items():
        if int(k) - 1000000 < 0:
            out_k = str(int(k) + 1000000)
            out_v = emb_dict[out_k]

            poincare_len = PoincareDistance(v,np.array([0.0,0.0]))
            euclidean_len = EuclideanDistance(v,np.array([0.0,0.0]))

            out_poincare_len = PoincareDistance(out_v,np.array([0.0,0.0]))
            out_euclidean_len = EuclideanDistance(out_v,np.array([0.0,0.0]))

            x_vals.append(
                (v[0]*poincare_len/euclidean_len + out_v[0]*out_poincare_len/out_euclidean_len)/2.0
            )
            y_vals.append(
                (v[1]*poincare_len/euclidean_len + out_v[1]*out_poincare_len/out_euclidean_len)/2.0
            )

            labels.append(code2names[str(k)]['name'])
            asymmetry = PoincareDistance(v,out_v)
            asymmetries.append(asymmetry)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    return x_vals, y_vals, asymmetries, labels

def get_map():
    prov_map = {
        '吉林省': 'Northeast China',
        '黑龙江省': 'Northeast China',
        '辽宁省': 'Northeast China',
        '新疆维吾尔自治区': 'Northwest China',
        '青海省': 'Northwest China',
        '宁夏回族自治区': 'Northwest China',
        '甘肃省': 'Northwest China',
        '陕西省': 'Northwest China',
        '内蒙古自治区': 'North China',
        '北京市': 'North China',
        '天津市': 'North China',
        '河北省': 'North China',
        '山西省': 'North China',
        '江苏省': 'East China',
        '上海市': 'East China',
        '安徽省': 'East China',
        '浙江省': 'East China',
        '福建省': 'East China',
        '江西省': 'East China',
        '山东省': 'East China',
        '河南省': 'Central China',
        '湖南省': 'Central China',
        '湖北省': 'Central China',
        '广东省': 'South China',
        '广西壮族自治区': 'South China',
        '海南省': 'South China',
        '西藏自治区': 'Southwest China',
        '重庆市': 'Southwest China',
        '四川省': 'Southwest China',
        '贵州省': 'Southwest China',
        '云南省': 'Southwest China'
    }
    
    color_map = {
        'Northeast China': {
            'prov': ['吉林省','黑龙江省','辽宁省']
        },
        'Northwest China': {
            'prov': ['新疆维吾尔自治区','青海省','宁夏回族自治区','甘肃省','陕西省']
        },
        'North China': {
            'prov': ['内蒙古自治区','北京市','天津市','河北省','山西省']
        },
        'East China': {
            'prov': ['江苏省','上海市','安徽省','浙江省','福建省','江西省','山东省']
        },
        'Central China': {
            'prov': ['河南省','湖南省','湖北省']
        },
        'South China': {
            'prov': ['广东省','广西壮族自治区','海南省']
        },
        'Southwest China': {
            'prov': ['西藏自治区','重庆市','四川省','贵州省','云南省']
        }
    }

    return prov_map, color_map

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

def main_size_rank(data_path, version):
    code_path = '../../datasets/code2name.csv'
    code2names, names2code, prov2idx = convert(code_path)

    with open(data_path, 'rb') as f:
        emb_dict = pickle.load(f)

    x_vals, y_vals, asymmetries, labels = getData_size(emb_dict, code2names)

    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(8):
        circle = plt.Circle((0, 0), i, color='black' if i == 7 else 'gray', fill=False, linewidth=0.5 if i == 7 else 0.2, zorder=1)
        ax.add_artist(circle)

    prov_map, color_map = get_map()
    city_candidate = get_candidate()
    candidate_map = get_candidate_map()

    palette = sns.color_palette("Set2", n_colors=len(color_map))
    region_colors = dict(zip(color_map.keys(), palette))

    asymmetries = np.array(asymmetries)
    ranks = rankdata(asymmetries, method='min').astype(float)
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    ranks = 1.0 - ranks

    texts = []
    added_regions = set()
    for i in range(len(labels)):
        name = labels[i]
        prov = code2names[names2code[name][0]]['prov']
        region = prov_map[prov]
        color = region_colors[region]

        is_candidate = name in city_candidate
        edge_color = 'black' if is_candidate else 'none'
        z = 3 if is_candidate else 2
        label_region = region if region not in added_regions and not is_candidate else None
        marker = '^' if is_candidate else 'o'

        size = 180 * ranks[i]
        ax.scatter(x_vals[i], y_vals[i], s=size, color=color, edgecolors=edge_color, alpha=0.8, label=label_region, zorder=z, marker=marker)

        if label_region:
            added_regions.add(region)

        if is_candidate:
            text = plt.text(x_vals[i], y_vals[i], candidate_map[name], fontsize=12, weight='semibold', ha='center', va='center', zorder=4)
            texts.append(text)

    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, force_text=0.5, expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    region_legend_items = sorted(region_colors.items(), key=lambda x: x[0])
    region_legend = [ax.scatter([], [], s=150, c=color, label=region) for region, color in region_legend_items]

    size_legend = []
    for rank_val in [1.0, 0.5, 0.0]:
        size = 180 * rank_val
        label = f"Asymmetry Top: {int((1 - rank_val) * 100)}%"
        size_legend.append(ax.scatter([], [], s=size, c='gray', alpha=0.6, label=label))

    handles = region_legend + size_legend
    labels = [h.get_label() for h in handles]

    legend = plt.legend(handles, labels, prop={'size': 13}, loc='upper right', frameon=True)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_facecolor('#f9f9f9')

    plt.tight_layout()
    plt.savefig(f'../../results/figures/visualization/Visualization_{version}.png', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",type=str)
    args = parser.parse_args()

    data_path = '../../log/v_' + args.version + '/embedding'

    main_size_rank(data_path,args.version)