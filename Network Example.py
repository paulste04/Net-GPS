import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Set a random seed for reproducibility
np.random.seed(7)

# Number of nodes
n = 6

# Generate a random symmetric binary adjacency matrix with no self-loops
A = np.triu((np.random.rand(n, n) > 0.6).astype(int), 1)
A = A + A.T  # Make it symmetric

# Create graph from adjacency matrix
G = nx.from_numpy_array(A)

# Generate spring layout for positioning
pos = nx.spring_layout(G, seed=7)

# Plot the graph
plt.figure(figsize=(6, 6))
nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors='black', node_size=700)
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_color='black')

#plt.title("Network Graph")
plt.axis('off')
plt.show()


####

# Re-import libraries after code execution environment reset
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_nodes = 100
p_er = 0.05  # Edge probability for Erdős-Rényi
m_sf = 2     # Number of edges to attach for scale-free (Barabási–Albert)

# Generate graphs
G_er = nx.erdos_renyi_graph(n=n_nodes, p=p_er, seed=42)
G_sf = nx.barabasi_albert_graph(n=n_nodes, m=m_sf, seed=42)

# Compute degrees
deg_er = [d for n, d in G_er.degree()]
deg_sf = [d for n, d in G_sf.degree()]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Erdős-Rényi graph
pos_er = nx.spring_layout(G_er, seed=42)
nx.draw(G_er, pos_er, node_size=50, ax=axs[0, 0], node_color='skyblue', edge_color='gray')
axs[0, 0].set_title("Erdős-Rényi Graph")

# Plot degree distribution of ER graph
axs[0, 1].hist(deg_er, bins=range(max(deg_er)+1), color='skyblue', edgecolor='black')
axs[0, 1].set_title("Erdős-Rényi Degree Distribution")
axs[0, 1].set_xlabel("Degree")
axs[0, 1].set_ylabel("Frequency")

# Plot Scale-Free graph
pos_sf = nx.spring_layout(G_sf, seed=42)
nx.draw(G_sf, pos_sf, node_size=50, ax=axs[1, 0], node_color='lightcoral', edge_color='gray')
axs[1, 0].set_title("Scale-Free Graph")

# Plot degree distribution of SF graph
axs[1, 1].hist(deg_sf, bins=range(max(deg_sf)+1), color='lightcoral', edgecolor='black')
axs[1, 1].set_title("Scale-Free Degree Distribution")
axs[1, 1].set_xlabel("Degree")
axs[1, 1].set_ylabel("Frequency")

plt.tight_layout(h_pad=3.0)
plt.show()

###

import networkx as nx
import matplotlib.pyplot as plt

# Create an undirected graph
G = nx.Graph()

# Add edges to represent the chain A - B - C
edges = [("C", "B"), ("B", "A")]
G.add_edges_from(edges)

# Define angled positions for the nodes
pos = {"A": (0, 0), "B": (1, 1), "C": (2, 0)}

# Draw the graph
plt.figure(figsize=(6, 3))
nx.draw(
    G, pos, with_labels=True,
    node_size=3000,
    node_color='white',     # white inner color
    edgecolors='black',     # black border
    linewidths=2,
    font_size=20,
    font_weight='bold',
    edge_color='gray'
)

plt.axis('off')  # Remove axes
plt.show()


####

# Re-run the complete code after reset

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import eigvals

# Set style
sns.set(style="whitegrid")

# Generate a denser scale-free graph (Barabási–Albert with more edges per new node)
G = nx.barabasi_albert_graph(n=12, m=4, seed=42)

# Convert to binary adjacency matrix
A = nx.to_numpy_array(G)
np.fill_diagonal(A, 0)
A[A > 0] = 1  # Ensure binary

# Row-normalized version
row_sums = A.sum(axis=1, keepdims=True)
W_row = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)

# Eigenvalue-normalized version
lambda_max = max(abs(eigvals(A)))
W_eig = A / lambda_max

# Plotting function
def plot_matrix(matrix, title, ax):
    sns.heatmap(matrix, annot=False, cmap='Blues', cbar=True, ax=ax, square=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# Plot the matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_matrix(A, 'Adjacency Matrix (Scale-free Network)', axes[0])
plot_matrix(W_row, 'Row-normalized Matrix', axes[1])
plot_matrix(W_eig, 'Eigenvalue-normalized Matrix', axes[2])

plt.tight_layout()
plt.show()
