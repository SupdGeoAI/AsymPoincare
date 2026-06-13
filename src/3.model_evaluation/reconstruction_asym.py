import pickle
import argparse
import numpy as np
import networkx as nx
from operator import itemgetter

fst = itemgetter(0)
snd = itemgetter(1)

def network_getTopN_in_out(g, N):
    dict_network_topN_in = {}
    dict_network_topN_out = {}
    nodes = g.nodes()
    for node in nodes:
        out_neighs = list(g.neighbors(node))
        out_weights = [g[node][neigh]['weight'] for neigh in out_neighs]
        out_sorted_neighs = [snd(tup) for tup in sorted(zip(out_weights, out_neighs), key=fst)][::-1][:N]
        dict_network_topN_out[str(node)] = [str(neigh) for neigh in out_sorted_neighs]

        in_neighs = list(g.predecessors(node))
        in_weights = [g[neigh][node]['weight'] for neigh in in_neighs]
        in_sorted_neighs = [snd(tup) for tup in sorted(zip(in_weights, in_neighs), key=fst)][::-1][:N]
        dict_network_topN_in[str(node)] = [str(neigh) for neigh in in_sorted_neighs]

    return dict_network_topN_in, dict_network_topN_out

def PoincareDistance(u,v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u-v)
    return float(np.arccosh(1+2*(euclidean_dist**2)/((1-norm_u**2)*(1-norm_v**2))))

def getDistanceMatrix(emb_array):
    size = emb_array.shape[0]
    matrix = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i,j] = PoincareDistance(emb_array[i],emb_array[j])
    return matrix

def embedding_getTopN_in(matrix, emb_idx2word, N):
    dict_topN = {}
    size = matrix.shape[0]
    for i in range(size):
        if int(emb_idx2word[i]) > 1000000:
            continue
        dist = matrix[i]
        idx = [j for j in range(size)]
        sorted_idx = sorted(idx, key=lambda j: dist[j])
        out_neighbors = []
        for j in sorted_idx:
            if int(emb_idx2word[j]) > 1000000:
                out_neighbors.append(str(int(emb_idx2word[j]) % 1000000))
            if len(out_neighbors) >= N:
                break
        dict_topN[str(emb_idx2word[i])] = out_neighbors[:N]    
    return dict_topN

def embedding_getTopN_out(matrix, emb_idx2word, N):
    dict_topN = {}
    size = matrix.shape[0]
    for i in range(size):
        if int(emb_idx2word[i]) < 1000000:
            continue
        dist = matrix[i]
        idx = [j for j in range(size)]
        sorted_idx = sorted(idx, key=lambda j: dist[j])
        in_neighbors = []
        for j in sorted_idx:
            if int(emb_idx2word[j]) < 1000000:
                in_neighbors.append(str(emb_idx2word[j]))
            if len(in_neighbors) >= N:
                break
        dict_topN[str(int(emb_idx2word[i]) % 1000000)] = in_neighbors[:N]
    return dict_topN

def calc_reconstruction_in_out(
    dict_network_topN_in,
    dict_network_topN_out,
    dict_embedding_topN_in,
    dict_embedding_topN_out,
    top_N
    ):

    reconstructions = []

    for node in dict_network_topN_in:
        true_in_neighbors = set(dict_network_topN_in[node])
        embedding_in_neighbors = set(dict_embedding_topN_in[node])
        intersection_in = true_in_neighbors & embedding_in_neighbors
        reconstruction_in = len(intersection_in) / top_N
        if reconstruction_in > 0:
            reconstructions.append(reconstruction_in)

        true_out_neighbors = set(dict_network_topN_out[node])
        embedding_out_neighbors = set(dict_embedding_topN_out[node])
        intersection_out = true_out_neighbors & embedding_out_neighbors
        reconstruction_out = len(intersection_out) / top_N
        if reconstruction_out > 0:
            reconstructions.append(reconstruction_out)

    mean_reconstruction = np.mean(reconstructions)
    print("Reconstruction@" + str(top_N) + ": " + str(np.mean(reconstructions)))

    return mean_reconstruction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--network_path", type=str)
    args = parser.parse_args()

    top_Ns = [i for i in range(10,41)]

    reconstructions = []
    for top_N in top_Ns:
        with open(args.network_path, 'rb') as f:
            network_data = pickle.load(f)
        graph = nx.DiGraph()
        graph.add_edges_from(network_data)
        dict_network_topN_in, dict_network_topN_out = network_getTopN_in_out(graph, top_N)

        data_path = '../../logs/v_' + args.version + "/embedding"
        with open(data_path, 'rb') as f:
            emb_data = pickle.load(f)
        emb_idx2word = []
        emb_array = []
        for k, v in emb_data.items():
            emb_idx2word.append(k)
            emb_array.append(v)
        emb_array = np.array(emb_array)
        distance_matrix = getDistanceMatrix(emb_array)
        dict_embedding_topN_in = embedding_getTopN_in(distance_matrix,emb_idx2word,top_N)
        dict_embedding_topN_out = embedding_getTopN_out(distance_matrix,emb_idx2word,top_N)
        
        reconstruction_avg = calc_reconstruction_in_out(
            dict_network_topN_in,
            dict_network_topN_out,
            dict_embedding_topN_in,
            dict_embedding_topN_out,
            top_N
        )
        
        reconstructions.append(reconstruction_avg)

    with open("../../results/stats/reconstruction/reconstruction_" + args.version, "wb") as f:
        pickle.dump(reconstructions, f)