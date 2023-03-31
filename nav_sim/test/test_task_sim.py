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

from nav_sim.env.task_env import TaskEnv


def run_env(task):
    env = TaskEnv(render=True)
    env.reset(task)

    # Press any key to start
    print("\n=========================================")
    input("Press any key to start")
    print("=========================================\n")

    # Run
    for step in range(100):

        # Execute action
        action = [0.1, 0.1, 0.01]
        observation, reward, done, info = env.step(action)

        # summarize the step in one line
        print(
            '\nStep: {}, Action: {}, Reward: {}, Done: {}, Info: {}\n'.format(
                step, action, reward, done, info
            )
        )

        # Show RGB image or LiDAR scan
        if task.observation.type == 'rgb':
            plt.imshow(observation.transpose(1, 2, 0))
            plt.show()
        elif task.observation.type == 'lidar':
            # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01
            observation = observation[:, observation[2, :] > 0.01]
            observation = observation[:, np.abs(observation[1, :]) < 3.5]
            observation = observation[:, observation[0, :] > 0.01]
            print('Scan - number of points: ', observation.shape[1])
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(
                observation[0, :], observation[1, :], observation[2, :]
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/allen/data/pac-perception/task.pkl',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # Sample random task
    task = random.choice(task_dataset)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Initialize task
    task.goal_radius = 0.5
    #
    task.observation = {}
    task.observation.type = 'lidar'  # 'rgb' or 'lidar'
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.rgb.x_offset_from_robot_front = 0.01  # no y offset
    task.observation.rgb.z_offset_from_robot_top = 0
    task.observation.rgb.tilt = 5  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 256
    task.observation.rgb.img_h = 256
    task.observation.rgb.aspect = 1
    task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
    task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
    task.observation.depth.img_h = task.observation.rgb.img_h
    task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
    task.observation.lidar.horizontal_res = 1  # resolution, in degree
    task.observation.lidar.vertical_res = 1  # resolution, in degree
    task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
    task.observation.lidar.max_range = 5  # in meter

    # Run environment
    run_env(task)
