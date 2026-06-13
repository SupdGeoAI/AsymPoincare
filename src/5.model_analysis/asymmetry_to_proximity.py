import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    return hierarchies, asymmetries, labels

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
            'color': '#B4D163',
            'prov': ['吉林省','黑龙江省','辽宁省']
        },
        'Northwest China': {
            'color': '#4DAEA5',
            'prov': ['新疆维吾尔自治区','青海省','宁夏回族自治区','甘肃省','陕西省']
        },
        'North China': {
            'color': '#FFEB4E',
            'prov': ['内蒙古自治区','北京市','天津市','河北省','山西省']
        },
        'East China': {
            'color': '#F2875B',
            'prov': ['江苏省','上海市','安徽省','浙江省','福建省','江西省','山东省']
        },
        'Central China': {
            'color': '#EC647A',
            'prov': ['河南省','湖南省','湖北省']
        },
        'South China': {
            'color': '#D3649E',
            'prov': ['广东省','广西壮族自治区','海南省']
        },
        'Southwest China': {
            'color': "#757FBC",
            'prov': ['西藏自治区','重庆市','四川省','贵州省','云南省']
        }
    }

    return prov_map, color_map

def plot_box_asymmetries(labels, asymmetries, code2names, version):
    prov_map, color_map = get_map()
    palette = sns.color_palette("Set2", n_colors=len(color_map))
    region_colors = dict(zip(color_map.keys(), palette))

    data = []
    for i in range(len(labels)):
        prov = code2names[labels[i]]['prov']
        region = prov_map[prov]
        data.append({'Region': region, 'Asymmetry': asymmetries[i]})
    df = pd.DataFrame(data)

    stats = df.groupby("Region")["Asymmetry"].agg(['mean', 'var']).reset_index()
    stats = stats.sort_values(by="mean")
    df["Region"] = pd.Categorical(df["Region"], categories=stats["Region"], ordered=True)

    unique_regions = stats["Region"].tolist()
    color_map_unified = {region: region_colors[region] for region in unique_regions}

    plt.figure(figsize=(13, 6))
    ax = sns.boxplot(
        x='Region',
        y='Asymmetry',
        data=df,
        showfliers=False,
        order=unique_regions,
        palette=color_map_unified,
        linewidth=1.2
    )

    tick_labels = [f"{region}" for i, region in enumerate(stats["Region"])]
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')
    plt.xlabel("Regions", fontsize=16)
    plt.ylabel("Asymmetries", fontsize=16)
    plt.ylim(0, df["Asymmetry"].max() + 0.004)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(f'../../results/figures/asymmetry_to_proximity/To_proximities_{version}.png', dpi=300)

def main(data_path, version):
    code_path = '../../datasets/code2name.csv'
    code2names, names2code, prov2idx = convert(code_path)

    with open(data_path,'rb') as f:
        emb_dict = pickle.load(f)
    
    hierarchies, asymmetries, labels = getData(emb_dict, code2names)
    plot_box_asymmetries(labels, asymmetries, code2names, version)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",type=str)
    args = parser.parse_args()

    data_path = '../../logs/v_' + args.version + '/embedding'
    main(data_path, args.version)