import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import build_model
from datasets import build_dataset

from itertools import product, combinations


from utils.pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep, pc_cam_to_3detr, is_inside_camera_fov
from utils.box_util import box2d_iou
from utils.make_args import make_args_parser

from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config

## Might want to move these parameter & model into main robot code
# camera + 3DETR
num_pc_points = 40000

parser = make_args_parser()
args = parser.parse_args(args=[])

# Dataset config: use SUNRGB-D
dataset_config = dataset_config()
# Build model
model, _ = build_model(args, dataset_config)

# Load pre-trained weights
sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
model.load_state_dict(sd["model"]) 

model = model.cuda()
model.eval()

device = torch.device("cuda")

visualize = False


def get_box(observation):
    observation = observation[:, observation[2, :] < 2.9] # remove too high
    observation = observation[:, observation[2, :] > 0.3] # remove too low

    visualize = False

    if (len(observation[0])>0):
        # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
        points_new = np.transpose(np.array(observation))
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
        points = preprocess_point_cloud(np.array(points_new), num_pc_points)
    else:
        # There are no returns from the LIDAR, object is not visible
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
    
    # Convert from camera frame to world frame
    point_clouds = []
    point_clouds.append(points)
    
    batch_size = 1
    pc = np.array(point_clouds).astype('float32')
    pc = pc.reshape((batch_size, num_pc_points, 3))

    pc_all = torch.from_numpy(pc).to(device)
    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    # start = tm.time()
    outputs = model(inputs)
    # end = tm.time()
    # print("Time taken for inference: ", end-start)
    
    bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
    cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()

    chair_prob = cls_prob[:,:,3]
    sort_box = torch.sort(chair_prob,1,descending=True)

    # Visualize
    if visualize:
        pc_plot = pc[:, pc[0,:,2] > 0.0,:]
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

    num_probs = 0
    num_boxes = 15
    corners = []
    for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
        if (num_probs < num_boxes):
            prob = prob.numpy()
            bbox = bbox_pred_points[range(batch_size), sorted_idx, :, :]
            cc = pc_to_axis_aligned_rep(bbox.numpy())
            flag = False
            if num_probs == 0:
                corners.append(cc)
                num_probs +=1
            else:
                for cc_keep in corners:
                    bb1 = (cc_keep[0,0,0],cc_keep[0,0,1],cc_keep[0,1,0],cc_keep[0,1,1])
                    bb2 = (cc[0,0,0],cc[0,0,1],cc[0,1,0],cc[0,1,1])
                    # Non-maximal supression, check if IoU more than some threshold to keep box
                    if(box2d_iou(bb1,bb2) > 0.1):
                        flag = True
                if not flag:    
                    corners.append(cc)
                    num_probs +=1

            if visualize:
                r0 = [cc[0,0, 0], cc[0,1, 0]]
                r1 = [cc[0,0, 1], cc[0,1, 1]]
                r2 = [cc[0,0, 2], cc[0,1, 2]]

                for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                        np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                        np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                        if (visualize and not flag):
                            ax.plot3D(*zip(s, e), color=(0.5+0.5*prob, 0.1,0.1))
    
    boxes = np.zeros((len(corners),2,2))
    for i in range(len(corners)):
        boxes[i,:,:] = corners[i][0,:,0:2]
        # boxes[i,0,:] = np.array([[[0,-1],[1,0]]])@boxes[i,0,:]+np.array([sp.world.w/2,0])
        # boxes[i,1,:] = np.array([[[0,-1],[1,0]]])@boxes[i,1,:]+np.array([sp.world.w/2,0])
        # boxes[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])

    return boxes