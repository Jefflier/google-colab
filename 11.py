import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import os

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

if __name__ == "__main__":
    L_list = list(range(1, 15))
    m_values = [4]
    k_values = np.linspace(0, 8, 600)

    kc_results = {}

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

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

        # 保存数据为 CSV
        df = pd.DataFrame({
            "L": L_list,
            "k_c": kc_list
        })
        df.to_csv(f"{output_dir}/kc_vs_L_m{m}.csv", index=False)

        # 绘图并保存
        plt.figure(figsize=(8, 5))
        plt.plot(L_list, kc_list, marker='o', label=f"$m={m}$")
        plt.xlabel("Tree depth $L$")
        plt.ylabel("Threshold $k_c$")
        plt.title(f"Threshold $k_c$ vs Tree Depth $L$ (m={m})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kc_vs_L_m{m}.png", dpi=300)
        plt.savefig(f"{output_dir}/kc_vs_L_m{m}.pdf")
        plt.close()

    print("\n所有图像和数据已保存在 results/ 目录。")
