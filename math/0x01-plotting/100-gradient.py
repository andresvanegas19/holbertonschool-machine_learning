#!/usr/bin/env python3
''' This script plot a grahpic '''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

fig, axes = plt.subplots(figsize=(12, 10))

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

plt.scatter(x, y, z, c=np.arange(len(x)), cmap='winter')
cbar = plt.colorbar()

cbar.set_label('elevation (m)')
axes.set_title('Mountain Elevation')
axes.set_xlabel('x coordinate (m)')
axes.set_ylabel('y coordinate (m)')

plt.show()
