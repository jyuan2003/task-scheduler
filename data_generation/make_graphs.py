import networkx as nx
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


from pathlib import Path
import os 
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_graph(G, config, save_dir):
    save_path = save_dir/f"{config['size']}_{config['density']}_{config['weight']}_{config['structure']}.txt"
    if save_path.exists():
        print(f"{save_path} already exists, skipping...")
        return 
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


def make_graph(config, seed=42, vis=False, save_dir=PROJECT_ROOT/"data_generation/raw_dataset"):
    if config["size"] == "small":
        n = 1000
    elif config["size"] == "medium":
        n = 10000
    else:
        n = 100000
    
    if config["weight"] == "low_var":
        w_dist = (4, 6)
    else:
        w_dist = (1, 10)

    if config["structure"] == "random": 
        if config["density"] == "sparse":
            m = n
        elif config["density"] == "medium":
            m = math.ceil(math.log2(n)) * n
        else:
            m = math.ceil(math.log2(n)) * n * 10

        G = nx.gnm_random_graph(n, m, seed)


    elif config["structure"] == "cluster":
        cluster_mu = 10
        cluster_var = 5
        if config["density"] == "sparse":
            p_in = 0.1
            p_out = 0.01

        elif config["density"] == "medium":
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
        save_graph(G, config, save_dir)

def make_dataset():
    size = ["small", "medium", "large"]
    density = ["sparse", "medium", "dense"]
    weight = ["low_var", "high_var"]
    structure = ["random", "cluster"]

    configs = list(product(size, density, weight, structure))

    for x in configs:
        config = {"size":x[0], "density":x[1], "weight":x[2], "structure":x[3]}
        make_graph(config)

    print("Done...")

if __name__ == "__main__":
    make_dataset()