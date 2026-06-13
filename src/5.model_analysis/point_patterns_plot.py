import math
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def angle_with_x_axis(x,y):
    theta_rad = math.atan2(y, x)
    theta_deg = math.degrees(theta_rad)
    return theta_deg if theta_deg >= 0 else theta_deg + 360

def PoincareDistance(u,v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u-v)
    return float(np.arccosh(1+2*(euclidean_dist**2)/((1-norm_u**2)*(1-norm_v**2))))

def getSymData(emb_dict):
    degrees = []
    hierarchies = []
    for k,v in emb_dict.items():
        degree = angle_with_x_axis(v[0],v[1])
        hierarchy = PoincareDistance(v,np.array([0.0,0.0]))
        if not np.isnan(hierarchy):
            degrees.append(degree)
            hierarchies.append(hierarchy)

    return degrees, hierarchies

def getAsymData(emb_dict):
    degrees = []
    hierarchies = []
    for k, v in emb_dict.items():
        kid = int(k)
        if kid >= 1000000:
            continue
        out_k = str(kid + 1000000)
        if out_k not in emb_dict:
            continue
        v_in = np.asarray(v, dtype=float)
        v_out = np.asarray(emb_dict[out_k], dtype=float)
        degree_in = angle_with_x_axis(v_in[0],v_in[1])
        degree_out = angle_with_x_axis(v_out[0],v_out[1])
        hierarchy_in = PoincareDistance(v_in,np.array([0.0,0.0]))
        hierarchy_out = PoincareDistance(v_out,np.array([0.0,0.0]))   
        if not np.isnan(hierarchy_in) and not np.isnan(hierarchy_out):
            degrees.append((degree_in + degree_out)/2.0)
            hierarchies.append((hierarchy_in + hierarchy_out)/2.0)
    return degrees, hierarchies

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
        K_d = 2 * count / (lambd * N)
        K_vals.append(K_d)

    return np.array(K_vals)

def simulate_CSR_K_single_run(N, d_values):
    sim_angles = np.random.uniform(0, 360, size=N)
    return circular_K_function_fast(sim_angles, d_values) - 2 * d_values

def pointpats_analysis(angles, n_simulations, version):
    d_values = np.linspace(1,90,90)
    k_vals = circular_K_function_fast(angles, d_values)
    l_vals = k_vals - 2 * d_values

    N_sim = len(angles)
    simulations = Parallel(n_jobs=-1)(
        delayed(simulate_CSR_K_single_run)(N_sim, d_values) 
        for _ in tqdm(range(n_simulations), desc="Simulating CSR")
    )
    simulations = np.array(simulations)
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(d_values,l_vals,linewidth=2,alpha=0.9)
    plt.fill_between(d_values,lower,upper,color='gray',alpha=0.2,label='CSR 95% CI')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlabel("Degrees",fontsize=16)
    plt.ylabel("L-functions",fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("../../results/figures/L-function/L-function_{}.png".format(version), dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",type=str)
    parser.add_argument("--asymmetric",action="store_true")
    parser.add_argument("--simulations",type=int,default=999)
    args = parser.parse_args()

    args.data_path = f"../../logs/v_{args.version}/embedding"
    with open(args.data_path,'rb') as f:
        emb_dict = pickle.load(f)

    if not args.asymmetric:
        degrees, hierarchies = getSymData(emb_dict)
        pointpats_analysis(np.array(degrees), args.simulations, args.version)
    else:
        degrees, hierarchies = getAsymData(emb_dict)
        pointpats_analysis(np.array(degrees), args.simulations, args.version)