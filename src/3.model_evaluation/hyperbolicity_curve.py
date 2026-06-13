import json
import argparse
import matplotlib.pyplot as plt

def load_json(data_path):
    with open(data_path,'r') as f:
        data = json.load(f)
    return data

def plot_data(data_3, data_10):
    x = list(range(len(data_3)))
    x = [i + 5 for i in x]
    plt.figure(figsize=(10, 5))
    plt.plot(x, data_3, label='March', linestyle='-')
    plt.plot(x, data_10, label='October', linestyle='--')
    plt.xlabel("Neighborhood")
    plt.ylabel("δ-Hyperbolicity")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(data_path_3, data_path_10):
    data_3 = load_json(data_path_3)
    data_10 = load_json(data_path_10)
    plot_data(data_3, data_10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_3",type=str,default='../../../results/stats/hyperbolicity/edge_3.json')
    parser.add_argument("--data_path_10",type=str,default='../../../results/stats/hyperbolicity/edge_10.json')
    args = parser.parse_args()

    main(args.data_path_3, args.data_path_10)