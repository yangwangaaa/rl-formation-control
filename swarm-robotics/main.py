import numpy as np
from math import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.neighbors = []

    def set_pos_from_polar(self, r, theta, z):
        self.x = r * cos(theta)
        self.y = r * sin(theta)
        self.z = z

    def add_neighbors(self, neighbors):
        if type(neighbors) is list:
            self.neighbors += neighbors
        else:
            self.neighbors += [neighbors]

    def pre_update(self, dt):
        # dpos/dt = sum_neighbors alpha (pos_neighbor - pos)
        # new_pos = old_pos + dt * sum_neighbors ...
        if len(self.neighbors) == 4:
            self.next_x = self.x + dt * sum([n.x - self.x for n in self.neighbors])
            self.next_y = self.y + dt * sum([n.y - self.y for n in self.neighbors])
            self.next_z = self.z + dt * sum([n.z - self.z for n in self.neighbors])

    def update(self):
        if len(self.neighbors) == 4:
            self.x = self.next_x
            self.y = self.next_y
            self.z = self.next_z

    def __repr__(self):
        r = sqrt(self.x*self.x + self.y*self.y)
        theta = atan(self.y/self.x)*180/pi
        return '(%.2f, %.2f, %.2f)' % (r, theta, self.z)
        # return '(r=%.2f, theta=%.2f, z=%.2f, neighbors=%d)' % \
        #     (r, theta, self.z, len(self.neighbors))




if __name__ == '__main__':
    r = 5
    theta_lst = [i*pi/8 for i in range(16)]

    agents = []
    for _ in range(len(z_lst)):
        agents.append([0] * len(theta_lst))

    # create agents
    for i, z in enumerate(z_lst):
        for j, theta in enumerate(theta_lst):
            a = Agent()
            a.set_pos_from_polar(5, theta, z)
            agents[i][j] = a

    # link neighbors
    for i in range(len(z_lst)):
        for j in range(len(theta_lst)):
            agents[i][j].add_neighbors(agents[i][j-1])
            if j + 1 < len(theta_lst):
                agents[i][j].add_neighbors(agents[i][j+1])
            else:
                agents[i][j].add_neighbors(agents[i][0])
            if i > 0:
                agents[i][j].add_neighbors(agents[i-1][j])
            if i < len(z_lst)-1:
                agents[i][j].add_neighbors(agents[i+1][j])

    # flatten list of agents
    agents = [a for a_lst in agents for a in a_lst]

    print(agents)
    print("-"*20)

    xs1 = [a.x for a in agents]
    ys1 = [a.y for a in agents]
    zs1 = [a.z for a in agents]

    # make a couple iterations
    dt = 1e-3
    for i in range(2):
        for a in agents:
            a.pre_update(dt)
        for a in agents:
            a.update()
        # print("iter %d" % i)
    print(agents)

    xs2 = [a.x for a in agents]
    ys2 = [a.y for a in agents]
    zs2 = [a.z for a in agents]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs1, ys1, zs1, marker='o')
    ax.scatter(xs2, ys2, zs2, marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plt.show()
