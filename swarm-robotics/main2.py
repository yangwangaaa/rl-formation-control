"""
N/M <-> w/h ?

origin = (0,0,0) ? (#30)


"""


import numpy as np
from math import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation


# parameters
N = 20  # 81
M = 20  # 90

w = N  # ?
h = M  # ?
r = 1


# build agents
agents = np.zeros((h, w, 3))

for i, theta in enumerate(np.linspace(0, 2 * pi, num=w, endpoint=False)):
    # agents[:, i, 0] = r * cos(3 * theta)
    # agents[:, i, 1] = r * sin(2 * theta)
    agents[:, i, 0] = r * cos(theta)
    agents[:, i, 1] = r * sin(theta)
    agents[1:-2, i, 0] = 3*r * cos(theta)
    agents[1:-2, i, 1] = 3*r * sin(theta)
for z in range(h):
    agents[z, :, 2] = z / (h - 1)


agents_saves = [np.copy(agents)] 


# update rule parameters
h_theta = 2 * pi / (N - 1)
h_s = 1 / (M - 1)
betas = np.array([0, 0, 0])
lambdas = np.array([1, 1, 1]) * 15

# simulation parameters
dt = 1e-4
n_iters = 1500

# pre-computations
h_theta2 = h_theta * h_theta
h_s2 = h_s * h_s
betas_hs = betas / (2 * h_s)


# simulation
for k in range(n_iters):
    new_agents = np.copy(agents)

    for z in range(1, h - 1):
        for i in range(w):
            # update rule for (x, y, z)
            new_agents[z, i] = agents[z, i] + dt * (
                (agents[z, (i + 1) % w] - 2 * agents[z, i] + agents[z, i - 1]) / h_theta2 +
                (agents[z + 1, i] - 2 * agents[z, i] + agents[z - 1, i]) / h_s2 +
                betas_hs * (agents[z, (i + 1) % w] - agents[z, i - 1]) +
                lambdas * agents[z, i]
            )

    if k % 100 == 0:
        print("Iter %d/%d (diff: %f)" % (k, n_iters, np.linalg.norm(agents - new_agents)))
        agents_saves.append(np.copy(agents))
    agents = np.copy(new_agents)


# plot agents
def update_graph(num):
    data = agents_saves[num]
    graph._offsets3d = (data[:, :, 0].flatten(), data[:, :, 1].flatten(), data[:, :, 2].flatten())
    title.set_text('Formation control on cylindrical surface (frame #{})'.format(num))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('Formation control on cylindrical surface')
if True:
    # ax.scatter(agents_saves[0][:, :, 0].flatten(), agents_saves[0][:, :, 1].flatten(), agents_saves[0][:, :, 2].flatten(), marker='o')
    ax.scatter(agents_saves[-1][:, :, 0].flatten(), agents_saves[-1][:, :, 1].flatten(), agents_saves[-1][:, :, 2].flatten(), marker='o')
graph = ax.scatter(agents_saves[0][:, :, 0].flatten(), agents_saves[0][:, :, 1].flatten(), agents_saves[0][:, :, 2].flatten(), marker='^')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(agents_saves),
                               interval=300, blit=False, repeat=True)

plt.show()
