#!/usr/bin/env python3
''' This script plot a grahpic '''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# lib =
data = np.load('./data-pca.npy')
labels = np.load('./labels-pca.npy')

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)


fig, axes = plt.subplots(figsize=(6, 6))
ax = Axes3D(fig)

ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    pca_data[:, 2],
    c=labels,
    cmap='prism',
    alpha=1
)

ax.set_title('PCA of Iris Dataset', size=15)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

plt.show()
