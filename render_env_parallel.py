'''
Load an object (chair) into an empty scene in a random location. 
Modified from iGibson/igibson/examples/objects/load_objects.py.
'''

# Things to do:

# Setup training; add residual style model to 3DETR decoder



import logging
import os
from sys import platform
import IPython as ipy
import time
from multiprocessing import Pool

import numpy as np
import itertools
import trimesh
import yaml

from pc_utils import write_ply_3DETR, write_ply_3DETR_rgb

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots.turtlebot import Turtlebot
# from igibson.robots.husky import Husky
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import let_user_pick, parse_config, quat_pos_to_mat


def render_env(seed):
    """
    Load an object (chair) into an empty scene in a random location and render different camera views.
    Coordinate system of scene: +X (right), +Y (forward), +Z (up)
    """

    ########################################
    # Define some parameters
    start_loc = np.array([0, -10, 0.5])
    camera_height = 0.5 
    view_direction = np.array([0, 1, 0])
    cam_dist_thresh = 5.0

    x_lims = [-2, 2]
    y_lims = [-10, -4]

    obs_x_lims = x_lims
    obs_y_lims = [y_lims[0]+1, y_lims[1]-1]

    x_grid = np.linspace(-2,2,10)
    y_grid = np.linspace(-10, -10+6, 20)
    ########################################


    ########################################
    headless=True
    ########################################


    ########################################
    np.random.seed(seed)
    ########################################


    ########################################

    ##### Define type of scene (empty) #####
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)
    ########################################


    ##### Viewer ###########################
    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [2, -10, 1.2] 
        s.viewer.initial_view_direction = [-0.5, 1, 0] 
        s.viewer.reset_viewer()
    ########################################


    ########################################
    # Load object randomly

    obj_x_loc = np.random.uniform(obs_x_lims[0], obs_x_lims[1])
    obj_y_loc = np.random.uniform(obs_y_lims[0], obs_y_lims[1])
    obj_z_loc = 0.5

    obj_yaw = np.random.uniform(-180, 180)

    objects_to_load = {
        "chair_1": {
            "category": "swivel_chair",
            "pos": (obj_x_loc, obj_y_loc, obj_z_loc),
            "orn": (0, 0, obj_yaw),
        },
    }

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_ig_avg_category_specs()

    scene_objects = {}
    try:
        for obj in objects_to_load.values():
            category = obj["category"]
            if category in scene_objects:
                scene_objects[category] += 1
            else:
                scene_objects[category] = 1

            # Get the path for all models of this category
            category_path = get_ig_category_path(category)

            # If the specific model is given, we use it. If not, we select one randomly
            if "model" in obj:
                model = obj["model"]
            else:
                model = np.random.choice(os.listdir(category_path))

            # Create the full path combining the path for all models and the name of the model
            model_path = get_ig_model_path(category, model)
            filename = os.path.join(model_path, model + ".urdf")

            # Create a unique name for the object instance
            obj_name = "{}_{}".format(category, scene_objects[category])

            # Create and import the object
            simulator_obj = URDFObject(
                filename,
                name=obj_name,
                category=category,
                model_path=model_path,
                avg_obj_dims=avg_category_spec.get(category),
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
            s.import_object(simulator_obj)
            simulator_obj.set_position_orientation(obj["pos"], quat_from_euler(obj["orn"]))


        ########################################

        ########################################
        # Simulate for some steps to let object settle
        max_steps = 20
        for i in range(max_steps):
            # with Profiler("Simulator step"):
            s.step()
        ########################################


        ########################################
        # Save ground truth bounding box
        bbox_center, bbox_orn, bbox_bf_extent, bbox_wf_extent = simulator_obj.get_base_aligned_bounding_box(visual=(not headless))
        
        # Get vertex positions in bbox reference frame
        bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (bbox_bf_extent / 2)

        # Convert to world frame
        bbox_transform = quat_pos_to_mat(bbox_center, bbox_orn)
        bbox_world_frame_vertex_positions = trimesh.transformations.transform_points(bbox_frame_vertex_positions, bbox_transform)
        ########################################


        ########################################
        # Take images from camera and save point clouds

        # Initialize data structures for results
        num_views = len(x_grid)*len(y_grid) # Total number of camera views
        cam_positions = num_views*[None]
        point_clouds = num_views*[None]

        ind = 0
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):

                # Camera location
                camera_pos = np.array([x_grid[i], y_grid[j], camera_height])
                cam_positions[ind] = camera_pos


                s.renderer.set_camera(camera_pos, camera_pos + view_direction, [0, 0, 1])
                s.renderer.set_fov(90)

                # rgb = s.renderer.render(modes=("rgb"))
                # colors = 255*rgb[0].reshape((rgb.shape[0]*rgb.shape[1], 4))[:,0:3]
                pc = s.renderer.render(modes=("3d"))[0]
                points = pc.reshape((pc.shape[0]*pc.shape[1],4))[:,0:3]

                # Get rid of points that are too far away
                inds_close = (points[:,2] > -cam_dist_thresh)

                points = points[inds_close, :]
                # colors = colors[inds_close, :]

                # write_ply_3DETR(points, "gibson.ply")
                # write_ply_3DETR_rgb(points, colors, "gibson.obj")

                # Save point cloud from this view
                point_clouds[ind] = points

                # Update index
                ind += 1

        ########################################

    finally:
        s.disconnect()


    return cam_positions, point_clouds, bbox_world_frame_vertex_positions


def main(raw_args=None):

    # Number of parallel threads
    num_parallel = 2

    # _, _ = render_env(seed=100)

    with Pool(num_parallel) as p:

        # Pack inputs to use with map
        inputs = range(num_parallel) # Seeds to use for each thread

        # Run in parallel
        outputs = p.map(render_env, inputs)

    # ipy.embed()

if __name__ == "__main__":
    main()