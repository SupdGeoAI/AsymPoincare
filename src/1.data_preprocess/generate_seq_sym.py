import pickle
import random
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm

def getTransitionMatrix(G, nodes, epsilon=1e-5):
    nodes2idx = {}
    count = 0
    for node in nodes:
        nodes2idx[node] = count
        count = count + 1

    size = len(nodes)
    matrix = np.zeros((size,size)) + 1e-5
    for i in tqdm(range(0,size),desc="Computing Transition Matrix"):
        out_edges = list(G.out_edges(nodes[i],data=True))
        for edge in out_edges:
            u,v,prop = edge
            matrix[i][nodes2idx[v]] = prop['weight']
    matrix = matrix/matrix.sum(axis=1)[:,np.newaxis]

    return matrix

def generateSequence(startIndex, transitionMatrix, path_length, alpha):
    result = [startIndex]
    current = startIndex
    for i in range(0,path_length):
        if random.random() < alpha:
            nextIndex = startIndex
        else:
            probs = transitionMatrix[current]
            nextIndex = np.random.choice(len(probs),1,p=probs)[0]
        result.append(nextIndex)
        current = nextIndex

    return result

def random_walk(G, num_paths, path_length, alpha):
    nodes = list(G.nodes())
    transitionMatrix = getTransitionMatrix(G, nodes)
    sentenceList = []
    size = len(nodes)
    for i in tqdm(range(0,size),desc="Generating Sequences"):
        for j in range(0,num_paths):
            indexList = generateSequence(i, transitionMatrix, path_length, alpha)
            sentence = [int(nodes[tmp]) for tmp in indexList]
            sentenceList.append(sentence)

    return sentenceList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_paths", type=int)
    parser.add_argument("--path_length", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.edge_path, 'rb') as f:
        edges = pickle.load(f)

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    sentenceList = random_walk(graph, args.num_paths, args.path_length, args.alpha)

    with open(args.save_path,'wb') as f:
        pickle.dump(sentenceList,f)