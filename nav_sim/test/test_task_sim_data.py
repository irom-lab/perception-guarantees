"""
Test the navigation simulation with the task dataset.

Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool
import math

from clustering import cluster
from nav_sim.env.task_env import TaskEnv
import sys
sys.path.append('../utils')
from utils.pc_util import preprocess_point_cloud, pc_cam_to_3detr, is_inside_camera_fov
import warnings
warnings.filterwarnings("error")
import IPython as ipy

def run_env(task):
    visualize = True
    verbose = 0
    # init_state = [1,-3,-np.pi/2]
    init_state = [0,-3.5,0]
    goal_loc = [7.5,3.5]
    task.init_state = [float(v) for v in init_state]
    task.goal_loc = [float(v) for v in goal_loc]
    env = TaskEnv(render=visualize)
    env.reset(task)
    camera_pos = []
    cam_not_inside_obs = []
    is_visible = []
    point_clouds = []
    num_pc_points = 40000

    # Press any key to start
    # print("\n=========================================")
    # input("Press any key to start")
    # print("=========================================\n")

    # Run
    # for step in range(16*4):
    for step in range(49):
        # Execute action
        # action = [0.1, 0.1, 0.01]
        if ((step)%7 == 0 and step >0):
            # at the edge of the grid, go sideways
            action = [0.0, 0.5, 0.0]
        else:
            # otherwise go to the top/bottom of the room from the bottom/top
            action = [(0.5)*(-1)**(math.floor(step/7)), 0.0, 0]
        # if ((step)%16 == 0 and step >0):
        #     # at the edge of the grid, go sideways
        #     print("Go sideways")
        #     action = [0.0, 1.0, np.pi/4]
        # elif ((step)%4 == 0 and step >0):
        #     # otherwise go to the top/bottom of the room from the bottom/top
        #     action = [(-1)**(math.floor(step/16))*1, 0.0, np.pi/4]
        # else:
        #     # otherwise turn
        #     action = [0.0, 0, np.pi/4]
        observation, reward, done, info = env.step(action)

        # # summarize the step in one line
        # print(
        #     '\nStep: {}, Action: {}, Reward: {}, Done: {}, Info: {}\n'.format(
        #         step, action, reward, done, info
        #     )
        # )
        task.observation.camera_pos[step] = [float(env.lidar_pos[0]), float(env.lidar_pos[1]), float(env.lidar_pos[2])]
        pos = task.observation.camera_pos[step]
        camera_pos.append(pos)
        # rng = task.observation.lidar.max_range
        not_inside_xyz = [[(0 if (pos[i]>obs[i]-1 and pos[i]<obs[3+i]+1) else 1) for i in range(3)] for obs in task.piece_bounds_all]
        gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in task.piece_bounds_all]
        task.observation.cam_not_inside_obs[step] = all([True if any(obs) == 1 else False for obs in not_inside_xyz])
        cam_not_inside_obs.append(task.observation.cam_not_inside_obs[step])
        # # is_vis = [1 if (np.linalg.norm(np.array(pos)-np.array(obs[0:3]),2)<rng and np.linalg.norm(np.array(pos)-np.array(obs[3:6]),2)<rng) else 0 for obs in task.piece_bounds_all]
        # is_vis = is_inside_camera_fov(info['state'], task.piece_bounds_all, task.observation.rgb.fov)
        # task.observation.is_visible[step] = all(is_vis)
        is_vis = [False]*len(task.piece_bounds_all)
        
        # Show RGB image or LiDAR scan
        # if task.observation.type == 'rgb':
        #     task.observation.rgb = observation.transpose(1, 2, 0)
        #     plt.imshow(observation.transpose(1, 2, 0))
        #     plt.show()
        # elif task.observation.type == 'lidar':
        if task.observation.type == 'lidar' or task.observation.type == 'rgb':
            # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01 and within a 1m distance of the robot
            observation = observation[:, observation[2, :] < 2.9]
            observation = observation[:, np.abs(observation[1, :]) < 3.9]
            observation = observation[:, observation[0, :] > 0.05]
            observation = observation[:, observation[0, :] < 7.95]
            # observation = observation[:, (observation[0,:]-pos[0])**2 + (observation[1,:]-pos[1])**2 > 1]

            X = observation[:, ((observation[0,:]-pos[0])**2 + (observation[1,:]-pos[1])**2)**0.5 > 1.25]
            X = X[:, ((X[0,:]-pos[0])**2 + (X[1,:]-pos[1])**2)**0.5 < 6.5]
            # if(len(X) > 0):
            #     print(min((np.arctan2(X[1,:]-pos[1], X[0,:]-pos[0]))*180/np.pi), max((np.arctan2(X[1,:]-pos[1], X[0,:]-pos[0]))*180/np.pi))
            #     # print(min((np.arctan((X[0,:]-pos[0])/(X[1,:]-pos[1])))*180/np.pi), max((np.arctan((X[0,:]-pos[0])/(X[1,:]-pos[1])))*180/np.pi))
            #     # print(np.min((np.arctan2(np.abs(X[0,:]-pos[0]), np.abs(X[1,:]-pos[1])))*180/np.pi), np.max((np.arctan2(np.abs(X[0,:]-pos[0]), np.abs(X[1,:]-pos[1])))*180/np.pi))
            X = X[:, (X[2, :] >0.1)]
            # X = X[:, math.atan2(X[1,:]-pos[1], X[0,:]-pos[0]) ]
            X = np.transpose(np.array(X))
            if(len(X) > 0):
                cluster_centers = cluster(X, visualize)
                for obs_idx, obs in enumerate(task.piece_bounds_all):
                    is_vis[obs_idx] = any([all([cc[i]>obs[i]+0.1 and cc[i]<obs[3+i]-0.1 for i in range(2)]) for cc in cluster_centers])
            task.observation.is_visible[step] = is_vis
            if visualize:
                # Print scan
                print('Scan - number of points: ', observation.shape[1])
                plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(
                    observation[0, :], observation[1, :], observation[2, :]
                )
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_aspect('equal')
                plt.show()
                # ipy.embed() 

            if (len(observation[0])>0):
                # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
                points_ = np.zeros_like(np.transpose(np.array(observation)).shape)
                points_ = pc_cam_to_3detr(np.transpose(np.array(observation)))
                points = np.zeros((1,num_pc_points, 3),dtype='float32')
                points = preprocess_point_cloud(np.array(points_), num_pc_points)
                # print(points.shape)
            else:
                # There are no returns from the LIDAR, object is not visible
                print("Trying again... ")
                points = np.zeros((1,num_pc_points, 3),dtype='float32')
                # print(points.shape)
                task.observation.is_visible[step] = [False]*len(task.piece_bounds_all)

            # Convert from camera frame to world frame
            is_visible.append(task.observation.is_visible[step])
            point_clouds.append(points)
            observation = tuple(map(tuple, observation))
            observation = [[float(observation[i][j]) for j in range(len(observation[i]))] for i in range(3)]
            task.observation.lidar[step] = observation
    # try:
    #     if verbose:
    #         print('LIDAR...', point_clouds, task.piece_bounds_all)
    #     return {"cam_positions": np.array(camera_pos), # Camera positions in Gibson world frame
    #         "cam_not_inside_obs": np.array(cam_not_inside_obs), # Booleans saying whether camera was inside obstacle for each location
    #         "is_visible": np.array(is_visible), # Booleans saying whether obstacle is visible from each location
    #         "point_clouds": np.array(point_clouds), # Point clouds in Gibson world frame
    #         "bbox_world_frame_vertices": np.array(gt_obs), # Bounding boxes in Gibson world frame
    #         "bbox_world_frame_aligned": np.array(gt_obs)} # Axis-aligned bounding box representation
    # except:
    #     print('Error, printing...', point_clouds, cam_not_inside_obs, is_visible, task.piece_bounds_all, task.observation.camera_pos)
    return {"cam_positions": np.array(camera_pos), # Camera positions in Gibson world frame
        "cam_not_inside_obs": np.array(cam_not_inside_obs), # Booleans saying whether camera was inside obstacle for each location
        "is_visible": np.array(is_visible), # Booleans saying whether obstacle is visible from each location
        "point_clouds": np.array(point_clouds), # Point clouds in Gibson world frame
        "bbox_world_frame_vertices": np.array(gt_obs), # Bounding boxes in Gibson world frame
        "bbox_world_frame_aligned": np.array(gt_obs)} # Axis-aligned bounding box representation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/anushri/Documents/Projects/data/perception-guarantees/task.npz',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # Sample random task
    save_tasks = []
    # ipy.embed()
    for task in task_dataset:

        # task = random.choice(task_dataset)

        # Initialize task
        task.goal_radius = 0.5
        #
        task.observation = {}
        task.observation.type = 'rgb'  # 'rgb' or 'lidar'
        task.observation.rgb = {}
        task.observation.depth = {}
        task.observation.lidar = {}
        task.observation.camera_pos = {}
        task.observation.cam_not_inside_obs = {}
        task.observation.is_visible = {}
        task.observation.rgb.x_offset_from_robot_front = 0.01  # no y offset
        task.observation.rgb.z_offset_from_robot_top = 0
        task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
        task.observation.rgb.img_w = 256
        task.observation.rgb.img_h = 256
        task.observation.rgb.aspect = 1
        task.observation.rgb.fov = 80  # in PyBullet, this is vertical field of view in degrees
        task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
        task.observation.depth.img_h = task.observation.rgb.img_h
        task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
        task.observation.lidar.horizontal_res = 1  # resolution, in degree,1
        task.observation.lidar.vertical_res = 1  # resolution, in degree , 1
        task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
        task.observation.lidar.max_range = 3 # in meter Anushri changed from 5 to 8

        # Run environment
        # run_env(task)
        # save_tasks += [task]
        # print(len(save_tasks))

    ##################################################################
    # Number of environments
    num_envs = 100

    # Number of parallel threads
    num_parallel = 1
    ##################################################################

    # _, _, _ = render_env(seed=0)

    ##################################################################
    env = 0
    batch_size = 10
    save_file = args.save_dataset
    t_start = time.time()
    for task in task_dataset:
        save_tasks += [task]
        env += 1 
        if env%batch_size == 0 and env >0:
            batch = math.floor(env/batch_size)
            print("Saving batch", str(batch))
            pool = Pool(num_parallel) # Number of parallel processes
            seeds = range(batch) # Seeds to use for the different processes
            # print(task.piece_id_all)
            results = pool.map_async(run_env, task_dataset[env-batch_size:env]) # Compute results
            pool.close()
            pool.join()
            results = results.get()
            t_end = time.time()
            # Save data
            file_batch = save_file[:-4] + str(batch) + ".npz"
            print(file_batch)
            np.savez_compressed(file_batch, data=results)
            print("Time to generate results: ", t_end - t_start)
            save_tasks = []
    ##################################################################

    ##################################################################
    # Save data
    # np.savez_compressed(args.save_dataset, data=results)
    ##################################################################


    # print(len(save_tasks))
    # args.save_path = '/home/anushri/Documents/Projects/data/perception-guarantees/task_with_lidar1.pkl'
    # with open(args.save_path, 'wb') as f:
    #     pickle.dump(save_tasks, f)

    ##################################################################
    # Save data
    # np.savez("/home/anushri/Documents/Projects/data/perception-guarantees/task_with_lidar.npz", data=save_tasks)
    ##################################################################

