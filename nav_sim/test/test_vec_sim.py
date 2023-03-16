"""
Test the vectorized navigation simulation.


Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import numpy as np
from omegaconf import OmegaConf

from nav_sim.env.vanilla_env import VanillaEnv
from nav_sim.env.vec_env import make_vec_envs


def main(args, task):
    venv = make_vec_envs(
        env_class=VanillaEnv,
        seed=args.seed,
        num_process=args.num_env,
        cpu_offset=args.cpu_offset,
        device=args.device,
    )
    obs_all = venv.reset([task for _ in range(args.num_env)])
    print(
        '\nReceived observation from all environments with shape:',
        obs_all.shape
    )


if __name__ == '__main__':
    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Args
    args = OmegaConf.create()
    args.seed = 42
    args.num_env = 10
    args.cpu_offset = 0
    args.device = 'cpu'

    # Initialize task
    task = OmegaConf.create()
    task.init_state = [1, 0.0, 0.0]  # x, y, yaw
    task.goal_loc = [7, 1.0]
    task.goal_radius = 0.5
    #
    task.furniture = {}
    task.furniture.piece_1 = {
        'path':
            os.path.join(
                nav_sim_path,
                'asset/sample_furniture/00a91a81-fc73-4625-8298-06ecd55b6aaa/raw_model.obj'
            ),
        'position': [4, 0.5, 0.0],
        'yaw': 0
    }
    task.furniture.piece_2 = {
        'path':
            os.path.join(
                nav_sim_path,
                'asset/sample_furniture/59e52283-361c-4b98-93e9-0abf42686924/raw_model.obj'
            ),
        'position': [6, -1, 0.0],
        'yaw': -np.pi / 2
    }
    #
    task.observation = {}
    task.observation.type = 'lidar'  # 'lidar'
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.rgb.x_offset_from_robot_front = 0.01  # no y offset
    task.observation.rgb.z_offset_from_robot_top = -0.01
    task.observation.rgb.tilt = 5  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 256
    task.observation.rgb.img_h = 256
    task.observation.rgb.aspect = 1
    task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
    task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
    task.observation.depth.img_h = task.observation.rgb.img_h
    task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
    task.observation.lidar.horizontal_res = 3  # resolution, in degree
    task.observation.lidar.vertical_res = 5  # resolution, in degree
    task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
    task.observation.lidar.max_range = 10  # in meter

    # Run
    main(args, task)
