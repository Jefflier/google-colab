import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool

def update(S, k, A):
    return np.prod((1 - np.exp(-k * S)) ** A, axis=1)

def create_bethe_tree_adj_matrix(L, n):
    G = nx.Graph()
    node_id = 0
    layers = [[node_id]]
    G.add_node(node_id)
    node_id += 1
    for l in range(1, L + 1):
        prev_layer = layers[-1]
        this_layer = []
        for parent in prev_layer:
            num_children = n - 1 if parent != 0 else n
            for _ in range(num_children):
                G.add_node(node_id)
                G.add_edge(parent, node_id)
                this_layer.append(node_id)
                node_id += 1
        layers.append(this_layer)
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    return A

def compute_kc_for_L(args):
    L, m, k_values = args
    A = create_bethe_tree_adj_matrix(L, m)
    N = A.shape[0]
    S_init = np.full(N, 0.9)
    for idx, k in enumerate(k_values):
        S = np.copy(S_init)
        for _ in range(500):
            S_new = update(S, k, A)
            if np.linalg.norm(S_new - S) < 1e-5:
                break
            S = S_new
        if S[0] > 1e-5:
            return (L, k)
    return (L, np.nan)

if __name__ == "__main__":  # 必须有
    L_list = list(range(1, 15))
    m_values = [4]
    k_values = np.linspace(0, 8, 600)

    kc_results = {}

    for m in m_values:
        print(f"\n==== m = {m} ====")
        args_list = [(L, m, k_values) for L in L_list]
        with Pool() as pool:
            results = pool.map(compute_kc_for_L, args_list)
        results.sort()
        kc_list = []
        for L, kc in results:
            kc_list.append(kc)
            print(f"L={L}, k_c={kc:.5f}")
        kc_results[m] = kc_list

    # 绘图
    plt.figure(figsize=(8, 5))
    for m in m_values:
        plt.plot(L_list, kc_results[m], marker='o', label=f"$m={m}$")
    plt.xlabel("Tree depth $L$")
    plt.ylabel("Threshold $k_c$")
    plt.title("Threshold $k_c$ vs Tree Depth $L$ for Bethe Trees")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
