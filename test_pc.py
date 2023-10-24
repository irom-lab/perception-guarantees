import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/anushri/Documents/Projects/data/perception-guarantees/task_with_lidar.npz", allow_pickle=True)
data = data["data"]


env = [9, 235, 269]
idx = [46, 11, 31]
for i in range(3):
	plt.figure()
	point_clouds = data[env[i]]["point_clouds"]
	pc = np.array(point_clouds)
	ax = plt.axes(projection='3d')
	ax.scatter3D(
	    pc[idx[i],0,:,0], pc[idx[i],0,:,1],pc[idx[i],0,:,2]
	)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show()