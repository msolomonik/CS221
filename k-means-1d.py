import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


data = np.array([0,2,10,12, 100, 101, 88.5])
centroids = np.array([0, 1, 2]) # clusters 0...(K-1) are index of centroids array
print(data)

# key = data index, value = centroids index
# nvmd, easier if I do
# key = cluster index
# value is a list of points (indices)
cluster_assignments = defaultdict(list)


def get_min_centroid(x):
    min_centroid = 0
    min_val = (x-centroids[0])**2

    for i, centroid in enumerate(centroids):
        if (x-centroid)**2 < min_val:
            min_val = (x-centroid)**2
            min_centroid = i
    
    return min_centroid
for epoch in range(0, 2):
    print("epoch", epoch)
    cluster_assignments = defaultdict(list)

    # Assign points to clusters
    for i in range(0, data.size):
        x = data[i]
        cluster_assignments[get_min_centroid(x)].append(i)
        print("x", x)
        print("get_min_centroid(x)", get_min_centroid(x))
        print("cluster_assignments", cluster_assignments)
    # Set new centroids
    for cluster, points in cluster_assignments.items():
        new_centroid = np.mean([data[i] for i in points])
        print("new_centroid", new_centroid)
        centroids[cluster] = new_centroid

# AI NOTICE: everything below is written by AI, everything above is hand-written
# Visualize clusters
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
fig, ax = plt.subplots(figsize=(8, 2))
for cluster, points in cluster_assignments.items():
    xs = [data[i] for i in points]
    ax.scatter(xs, [0] * len(xs), c=colors[cluster % len(colors)],
              label=f'Cluster {cluster}', s=100, zorder=3)
ax.scatter(centroids, [0] * len(centroids), c='black', marker='X',
           s=200, zorder=4, label='Centroids')
ax.set_yticks([])
ax.set_xlabel('Value')
ax.set_title('K-Means 1D Clustering')
ax.legend()
plt.tight_layout()
plt.show()
