import math
import json
import random
import pickle
import argparse
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm
from numpy import log, sqrt

def symmetrize_edges(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G_undirected = nx.Graph()
    for u, v in G.edges():
        weight_uv = G[u][v]['weight']
        weight_vu = G[v][u]['weight'] if G.has_edge(v,u) else 0
        avg_weight = (weight_uv + weight_vu) / 2
        G_undirected.add_edge(u, v, weight=avg_weight)    
    symmetrized_edges = [(u, v, {'weight': G_undirected[u][v]['weight']}) for u, v in G_undirected.edges()]
    return symmetrized_edges, list(range(len(G_undirected.nodes())))

def generate_samples(id_list, ratio):
    total_combinations = math.comb(len(id_list), 4)
    num_samples = int(total_combinations * ratio)
    four_tuples = set()

    while len(four_tuples) < num_samples:
        sample = tuple(sorted(random.sample(id_list, 4)))
        four_tuples.add(sample)

    return four_tuples

def edges_selection(edges, neighborhood):
    G_undirected = nx.Graph()
    G_undirected.add_edges_from(edges)
    filtered_edges = []
    for node in G_undirected.nodes:
        edges = [(u, v, G_undirected[u][v]['weight']) for u, v in G_undirected.edges(node)]
        edges.sort(key=lambda x: x[2], reverse=True)
        filtered_edges.extend([(u, v, {'weight': prop}) for u, v, prop in edges[:neighborhood]])
    return filtered_edges

def compute_distance_matrix(edges):
    flow_matrix = {}
    total_flow = {}
    for edge in edges:
        i = str(edge[0])
        j = str(edge[1])
        flow = float(edge[2]['weight'])

        if i != j:
            if (i, j) not in flow_matrix and (j, i) not in flow_matrix:
                flow_matrix[(i, j)] = flow
                flow_matrix[(j, i)] = flow
            elif (i, j) in flow_matrix and (j, i) in flow_matrix:
                flow_matrix[(i, j)] = (flow + flow_matrix[(i, j)]) / 2
                flow_matrix[(j, i)] = flow_matrix[(i, j)]
            
            total_flow[i] = total_flow.get(i, 0) + flow
            total_flow[j] = total_flow.get(j, 0) + flow

    nodes = list(total_flow.keys())
    distance_matrix = np.zeros((len(nodes), len(nodes)))

    for idx_i, i in enumerate(nodes):
        for idx_j, j in enumerate(nodes):
            if i != j and (i, j) in flow_matrix:
                Xi = total_flow[i]
                Xj = total_flow[j]
                Xij = flow_matrix[(i, j)]
                distance_matrix[idx_i, idx_j] = (Xi * Xj) / Xij

    max_distance = np.max(distance_matrix)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j and distance_matrix[i, j] == 0:
                distance_matrix[i, j] = max_distance

    return distance_matrix

def process_samples(four_tuples, distance_matrix):
    d_avg, count_d_avg = 0.0, 0
    delta_avg, count_delta_avg = 0.0, 0

    for four_tuple in tqdm(four_tuples):
        x, y, v, w = four_tuple
        sums = []
        sums.append(log(distance_matrix[x, y]) + log(distance_matrix[v, w]))
        sums.append(log(distance_matrix[x, v]) + log(distance_matrix[y, w]))
        sums.append(log(distance_matrix[x, w]) + log(distance_matrix[y, v]))
        sums = sorted(sums)
        d_avg += sums[0] + sums[1] + sums[2]
        delta_avg += float(sums[2] - sums[1]) / 2
        count_d_avg += 6
        count_delta_avg += 1
    
    d_avg = d_avg / count_d_avg
    delta_avg = delta_avg / count_delta_avg

    return 2 * delta_avg / d_avg

def hyperbolicity_curve(data_path, output_file):
    with open(data_path,'rb') as f:
        edges = pickle.load(f)
    
    reserved_edges = []
    for edge in edges:
        if edge[2]['weight'] != 0:
            reserved_edges.append((str(edge[0]),str(edge[1]),{'weight': float(edge[2]['weight'])}))

    symmetrized_edges, id_list = symmetrize_edges(reserved_edges)
    four_tuples = generate_samples(id_list, 0.01)

    results = []
    for neighbor_size in tqdm(range(5,51)):
        left_edges = edges_selection(symmetrized_edges, neighbor_size)
        distance_matrix = compute_distance_matrix(left_edges)
        result = process_samples(four_tuples, distance_matrix)
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--save_path",type=str)
    args = parser.parse_args()

    hyperbolicity_curve(args.data_path, args.save_path)