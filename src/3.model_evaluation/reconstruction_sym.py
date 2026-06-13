import pickle
import argparse
import numpy as np
import networkx as nx
from operator import itemgetter

fst = itemgetter(0)
snd = itemgetter(1)

def network_getTopN(g, N):
    dict_topN = {}
    nodes = g.nodes()
    for node in nodes:
        neighs = list(g.neighbors(node))
        weights = []
        for neigh in neighs:
            weights.append(g[node][neigh]['weight'])
        neighs = [snd(tup) for tup in sorted(zip(weights, neighs), key=fst)][::-1]  
        neighs = [str(neigh) for neigh in neighs]
        dict_topN[str(node)] = neighs[:N]
    return dict_topN

def EuclideanDistance(u, v):
    return float(np.linalg.norm(u-v))

def PoincareDistance(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u - v)
    return float(np.arccosh(1 + 2 * (euclidean_dist ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2))))

def getDistanceMatrix(emb_array, space):
    size = emb_array.shape[0]
    matrix = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if i != j:
                if space == 'poincare':
                    matrix[i,j] = PoincareDistance(emb_array[i],emb_array[j])
                elif space == 'euclidean':
                    matrix[i,j] = EuclideanDistance(emb_array[i],emb_array[j])
    return matrix

def embedding_getTopN(matrix, emb_idx2word, N):
    dict_topN = {}
    size = matrix.shape[0]
    for i in range(size):
        dist = matrix[i]
        idx = [j for j in range(size)]
        idx = [snd(tup) for tup in sorted(zip(dist, idx), key=fst)]
        idx = idx[:N]
        dict_topN[str(emb_idx2word[i])] = [str(emb_idx2word[j]) for j in idx]
    return dict_topN

def calc_reconstruction(dict_network_topN, dict_embedding_topN, top_N):
    reconstructions = []
    for k, v in dict_network_topN.items():
        embedding_topN = dict_embedding_topN[k]
        v = set(v)
        embedding_topN = set(embedding_topN)
        intersection = v & embedding_topN
        reconstruction = len(intersection) / top_N
        if reconstruction > 0:
            reconstructions.append(reconstruction)
    print("Reconstruction@" + str(top_N) + ": " + str(np.mean(reconstructions)))
    return np.mean(reconstructions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--space",type=str)
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
        dict_network_topN = network_getTopN(graph, top_N)

        data_path = '../../logs/v_' + args.version + '/embedding'
        with open(data_path, 'rb') as f:
            emb_data = pickle.load(f)
        emb_idx2word = []
        emb_array = []
        for k, v in emb_data.items():
            emb_idx2word.append(k)
            emb_array.append(v)
        emb_array = np.array(emb_array)
        distance_matrix = getDistanceMatrix(emb_array,args.space)

        dict_embedding_topN = embedding_getTopN(distance_matrix, emb_idx2word, top_N)
        reconstruction_result = calc_reconstruction(dict_network_topN, dict_embedding_topN, top_N)
        reconstructions.append(reconstruction_result)
    
    with open("../../results/stats/reconstruction/reconstruction_" + args.version, "wb") as f:
        pickle.dump(reconstructions, f)