'''
With this simple program you can easily observe the metric tensor
mechanics on vectors that forms the unit-ellipse
'''


import numpy as np
import matplotlib.pyplot as plt

def metric_ellipse(g, num_points=200):
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = []
    
    for theta in angles:
        # For each angle calculate vector and perform double tensor folding
        v = np.array([np.cos(theta), np.sin(theta)])
        scale = np.sqrt(np.einsum('i,ij,j', v, g, v))   # Double tensor folding itself
        v_scaled = v / scale
        points.append(v_scaled)

    return np.array(points)

# Define metric tensors for euclidian (flat) and non-euclidian (curved) spaces
euclidian_g = np.array([[1.0, 0.],
                        [0., 1.0]])

noneuclidian_g = np.array([[1.0, 0.5],
                           [0.5, 2.0]])

ellipse = metric_ellipse(euclidian_g)
ellipse_in_curved_space = metric_ellipse(noneuclidian_g)

fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].plot(ellipse[:, 0], ellipse[:, 1])
ax[0].axhline(0, color='gray', linestyle='--')
ax[0].axvline(0, color='gray', linestyle='--')
ax[0].set_aspect('equal')
ax[0].set_title('Ellipse in euclidian space')
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)

ax[1].plot(ellipse_in_curved_space[:, 0], ellipse_in_curved_space[:, 1])
ax[1].axhline(0, color='gray', linestyle='--')
ax[1].axvline(0, color='gray', linestyle='--')
ax[1].set_aspect('equal')
ax[1].set_title('Ellipse in non-euclidian (curved) space')
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)

plt.show()
