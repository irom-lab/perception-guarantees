import os
import sys
# import torch

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement

# Mesh IO
import trimesh

import matplotlib.pyplot as pyplot

def read_ply_realsense(filename):
    """ read XYZ point cloud from filename PLY file from RealSense """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    # pc_array = np.array([[x, y, z, r, g, b] for x,y,z,r,g,b in pc])
    # pc_array = np.array([[x, y, z] for x,y,z in pc])

    # Change to correct coordinate system:
    # Real-Sense: +X (right), +Y (down),    +Z (forward) [But export to ply is different; correction below]
    # VoteNet:    +X (right),   +Y (forward), +Z (up) 
    pc_array = np.array([[x, -z, y, r, g, b] for x,y,z,r,g,b in pc])

    write_ply_rgb(pc_array[:,0:3], pc_array[:,3:6], "demo_files/input_pc_realsense_transformed.obj")

    pc_array = pc_array[:,0:3] 


    return pc_array

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def write_ply_3DETR(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    # Change to correct coordinate system:
    # iGibson: +X (right), +Y (up),    +Z (backward)
    # 3DETR:    +X (right),   +Y (forward), +Z (up) 
    points = [(points[i,0], -points[i,2], points[i,1]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_3DETR_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0], -points[i,2], points[i,1],c[0],c[1],c[2]))
    fout.close()