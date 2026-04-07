import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# 2D data: 3 natural clusters
data = np.array([
    # Cluster around (2, 3)
    [1.5, 2.8], [2.2, 3.5], [2.8, 2.5], [1.8, 3.2], [2.5, 3.0],
    # Cluster around (8, 8)
    [7.5, 8.2], [8.3, 7.8], [8.0, 8.5], [7.8, 7.5], [8.5, 8.8],
    # Cluster around (9, 2)
    [8.5, 1.5], [9.2, 2.3], [9.5, 1.8], [8.8, 2.5], [9.0, 2.0],
])
K = 3
np.random.seed(1)
initial_indices = np.random.choice(data.shape[0], K, replace=False)
centroids = data[initial_indices]
print(data)
print(data.shape[0])
print(data.shape[1])


# key = data index, value = centroids index
# nvmd, easier if I do
# key = cluster index
# value is a list of points (indices)
cluster_assignments = defaultdict(list)


def get_min_centroid(x):
    min_centroid = 0
    min_val = np.linalg.norm(x-centroids[0])**2
    # print("min_val", min_val)

    for i, centroid in enumerate(centroids):
        # print("(x-centroid)**2", np.linalg.norm(x-centroid)**2)
        # print("min_val", min_val)
        if np.linalg.norm(x-centroid)**2 < min_val:
            min_val = np.linalg.norm(x-centroid)**2
            min_centroid = i
    
    return min_centroid
for epoch in range(0, 5):
    print("epoch", epoch)
    cluster_assignments = defaultdict(list)

    # Assign points to clusters
    for i in range(0, data.shape[0]):
        x = data[i]
        # print("x", x)
        # print("get_min_centroid(x)", get_min_centroid(x))
        cluster_assignments[get_min_centroid(x)].append(i)
        # print("cluster_assignments", cluster_assignments)


    # Set new centroids
    for cluster, points in cluster_assignments.items():
        new_centroid = np.mean([data[i] for i in points], axis=0)
        # print("new_centroid", new_centroid)
        centroids[cluster] = new_centroid

print(centroids)
print(cluster_assignments)

# Visualize clusters
# # AI NOTICE: everything below is written by AI, everything above is hand-written
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
fig, ax = plt.subplots(figsize=(8, 6))
for cluster, points in cluster_assignments.items():
    pts = data[points]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors[cluster % len(colors)],
               label=f'Cluster {cluster}', s=80, zorder=3)
ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X',
           s=200, zorder=4, label='Centroids')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('K-Means 2D Clustering')
ax.legend()
plt.tight_layout()
plt.show()