import networkx as nx
import math
import random
import matplotlib.pyplot as plt
from itertools import product

import argparse

from pathlib import Path
import os 
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_graph(G, args, save_dir):
    save_path = save_dir/f"{args.nodes}_{args.density}_{args.weight}_{args.structure}.txt"
    if save_path.exists():
        print(f"{save_path} already exists, skipping...")
        return save_path
    os.makedirs(save_dir, exist_ok=True)

    n, m = G.number_of_nodes(), G.number_of_edges()
    with open(save_path, "w") as f:
        f.write(f"NODES {n}\n")
        for i in G.nodes():
            f.write(f"{i} {G.nodes[i]['weight']}\n")

        f.write(f"EDGES {m}\n")
        for u, v in G.edges():
            f.write(f"{u} {v} {G.edges[u, v]['weight']}\n")

    print(f"Saved graph to {save_path}...")
    return save_path

def make_graph(args, seed=42, vis=False, save_dir=PROJECT_ROOT/"data_generation/raw_dataset"):
    n = args.nodes
    
    if args.weight == "low_var":
        w_dist = (4, 6)
    else:
        w_dist = (1, 10)

    if args.structure == "random": 
        if args.density == "sparse":
            m = n
        elif args.density == "medium":
            m = math.ceil(math.log2(n)) * n
        else:
            m = math.ceil(math.log2(n)) * n * 10

        G = nx.gnm_random_graph(n, m, seed)


    elif args.structure == "cluster":
        cluster_mu = 10
        cluster_var = 5
        if args.density == "sparse":
            p_in = 0.1
            p_out = 0.01

        elif args.density == "medium":
           p_in = 0.2
           p_out = 0.02

        else:
            p_in = 0.3
            p_out = 0.03
           
        G = nx.gaussian_random_partition_graph(n, cluster_mu, cluster_var, p_in, p_out, seed)


    for i in G.nodes:
        G.nodes[i]["weight"] = random.uniform(*w_dist)
    
    for i, (u, v) in enumerate(G.edges()):
        G.edges[u, v]["weight"] = random.uniform(*w_dist)

    if vis:
        nx.draw(G, with_labels=True, font_weight="bold")
        plt.show()

    if save_dir:
        return save_graph(G, args, save_dir)

def make_dataset(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--density", choices=["sparse", "medium", "dense"], required=True)
    parser.add_argument("--weight", choices=["low_var", "high_var"], required=True)
    parser.add_argument("--structure", choices=["random", "cluster"], required=True)
    parser.add_argument("--save_path", type=Path)
    args = parser.parse_args(argv)

    save_path = make_graph(args)

    print("Done...")

    return save_path 

if __name__ == "__main__":
    make_dataset()