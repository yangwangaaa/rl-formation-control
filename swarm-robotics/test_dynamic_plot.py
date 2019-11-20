import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from matplotlib.patches import Circle 


a = np.random.rand(2000, 3)*10
t = np.array([np.ones(100)*i for i in range(20)]).flatten()
df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})

def update_graph(num):
    print(graph._offsets3d[0].shape)
    graph._offsets3d[0] += [0.1] * 100
    graph._offsets3d = (data.x, data.y, data.z)
    # graph._facecolor3d = [[1, 0, 1, 1]]
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('Swarm robotics simulation')

data=df[df['time']==0]
print(data)
graph = ax.scatter(data.x, data.y, data.z, c="blue")

# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html
ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames=100, 
                               interval=100, repeat=False)

plt.show()

# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html