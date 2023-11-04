""" 
Compute feature representation from point clouds using pre-trained 3DETR. 
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import IPython as ipy
import json
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from models import build_model
from datasets import build_dataset
import matplotlib.pyplot as plt


from itertools import product, combinations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pretrained'))

from pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep
from box_util import box2d_iou
from make_args import make_args_parser


if __name__=='__main__':


    ###########################################################################
    # Parse arguments
    parser = make_args_parser()
    args = parser.parse_args()

    # Dataset config: use SUNRGB-D
    from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
    dataset_config = dataset_config()

    # Build model
    model, _ = build_model(args, dataset_config)

    # Load pre-trained weights
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
    model.load_state_dict(sd["model"]) 

    model = model.cuda()
    model.eval()

    device = torch.device("cuda")

    visualize = True
    ###########################################################################

    ###########################################################################
    # Load params from json file
    with open("env_params.json", "r") as read_file:
        params = json.load(read_file)
    ###########################################################################


    ###########################################################################
    # Get data
    # data = np.load("data/training_data_raw.npz", allow_pickle=True)
    # data = np.load("third_party/mmdetection3d-main/data/s3dis/s3dis_infos_Area_1.pkl", allow_pickle=True)
    # data = np.load("/home/anushri/Documents/Projects/data/perception-guarantees/task_with_lidar500rooms-floor.npz", allow_pickle=True)
    # filename = "/home/anushri/Documents/Projects/data/perception-guarantees/multiple_chairs_dataset/task_camera.npz"
    filename = "home/zm2074/Projects/data/perception-guarantees/"
    num_files = 10
    for i in range(num_files):
        file_ = filename[:-4] + str(i+1) + ".npz"
        data_ = np.load(file_, allow_pickle=True)
        if i == 0:
            data = data_["data"]
        else:
            data = np.concatenate((data, data_["data"]), axis=0)
    # data = data["data"]

    num_envs = len(data) # Number of environments where we collected data
    num_chairs = len(data[0]["bbox_world_frame_vertices"])
    num_cam_positions = len(data[0]['cam_positions']) # Number of camera locations in this environment
    num_boxes= 10
    print(num_cam_positions)

    num_pc_points = params["num_pc_points"] # Number of points in each point cloud

    # Batch size for processing inputs on GPU
    batch_size = 1 # Cannot be too large since GPU runs out of memory, 2

    num_batches = int(num_cam_positions / batch_size)
    assert (num_cam_positions % batch_size) == 0, "batch_size must divide num_cam_positions."
    ###########################################################################

    ###########################################################################


    ###########################################################################
    # Initialize data structure for ground truth bounding boxes for each environment
    #######
    bboxes_ground_truth = torch.zeros(num_envs, num_chairs, 2, 3)
    bboxes_ground_truth_aligned = torch.zeros(num_envs, num_chairs, 2,3)

    # bboxes_ground_truth = torch.zeros(num_envs, 8, 3)
    # bboxes_ground_truth_aligned = torch.zeros(num_envs, 2, 3)

    # Initialize loss mask array
    loss_mask = torch.zeros(num_envs, num_cam_positions, num_chairs)


    # Initialize outputs
    model_outputs_all = {
        "sem_cls_logits": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_semcls+1),
        "center_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "center_unnormalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "size_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "size_unnormalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "angle_logits": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_residual": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_residual_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_continuous": torch.zeros(num_envs, num_cam_positions, args.nqueries),
        "objectness_prob": torch.zeros(num_envs, num_cam_positions, args.nqueries),
        "sem_cls_prob": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_semcls),
        "box_corners": torch.zeros(num_envs, num_cam_positions, args.nqueries, 8, 3),
        "box_features": torch.zeros(num_envs, num_cam_positions, args.nqueries, args.dec_dim),
        "box_axis_aligned": torch.zeros(num_envs, num_cam_positions, num_chairs, 2, 3),
    }
    ###########################################################################

    ###########################################################################
    most_likely_boxes = torch.zeros(num_envs, num_cam_positions, num_boxes, 2, 3)
    t_start = time.time()
    for env in range(num_envs):
    # for env in [8]:

        print("Env: ", env)

        #####################################################
        # Save ground truth bounding box for this environment
        # bboxes_ground_truth[env,:,:] = torch.squeeze(torch.tensor(data[env]["bbox_world_frame_vertices"]),0)
        # bboxes_ground_truth_aligned[env,:,:] =torch.squeeze(torch.tensor(data[env]["bbox_world_frame_aligned"]),0)
        bboxes_ground_truth[env,:,:,:] = torch.tensor(data[env]["bbox_world_frame_vertices"])
        bboxes_ground_truth_aligned[env,:, :,:] = torch.tensor(data[env]["bbox_world_frame_aligned"])
        #####################################################

        for i in range(num_batches):
        # for i in [43]:
            # Read point clouds
            # if i == 46:
            #     visualize = True
            batch_inds = slice(i*batch_size, (i+1)*batch_size)
            point_clouds = data[env]["point_clouds"][batch_inds]
            pc = np.array(point_clouds).astype('float32')
            pc = pc.reshape((batch_size, num_pc_points, 3))
            pc_all = torch.from_numpy(pc).to(device)
            # pc = pc[batch_inds]
            # # pc = pc.reshape((batch_size, num_pc_points, 3))
            # pc = pc.T
            pc_min_all = pc_all.min(1).values
            pc_max_all = pc_all.max(1).values
            inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}
            pos = data[env]["cam_positions"][batch_inds][0]

            # Run through pre-trained 3DETR model
            try:
                outputs = model(inputs)
                # ipy.embed()
            except:True

            #####################################
            # Save outputs from model
            model_outputs_all["sem_cls_logits"][env,batch_inds,:,:] = outputs["outputs"]["sem_cls_logits"].detach().cpu()
            model_outputs_all["center_normalized"][env,batch_inds,:,:] = outputs["outputs"]["center_normalized"].detach().cpu()
            model_outputs_all["center_unnormalized"][env,batch_inds,:,:] = outputs["outputs"]["center_unnormalized"].detach().cpu()
            model_outputs_all["size_normalized"][env,batch_inds,:,:] = outputs["outputs"]["size_normalized"].detach().cpu()
            model_outputs_all["size_unnormalized"][env,batch_inds,:,:] = outputs["outputs"]["size_unnormalized"].detach().cpu()
            model_outputs_all["angle_logits"][env,batch_inds,:,:] = outputs["outputs"]["angle_logits"].detach().cpu()
            model_outputs_all["angle_residual"][env,batch_inds,:,:] = outputs["outputs"]["angle_residual"].detach().cpu()
            model_outputs_all["angle_residual_normalized"][env,batch_inds,:,:] = outputs["outputs"]["angle_residual_normalized"].detach().cpu()
            model_outputs_all["angle_continuous"][env,batch_inds,:] = outputs["outputs"]["angle_continuous"].detach().cpu()
            model_outputs_all["objectness_prob"][env,batch_inds,:] = outputs["outputs"]["objectness_prob"].detach().cpu()
            model_outputs_all["sem_cls_prob"][env,batch_inds,:,:] = outputs["outputs"]["sem_cls_prob"].detach().cpu()
            model_outputs_all["box_corners"][env,batch_inds,:,:] = outputs["outputs"]["box_corners"].detach().cpu()
            model_outputs_all["box_features"][env,batch_inds,:,:] = outputs["box_features"].detach().cpu()

            # Compute axis-aligned bbox representation of most prominent bbox from prediction
            bbox_pred_points = outputs["outputs"]["box_corners"].detach().cpu()
            if np.any(np.isnan(np.array(bbox_pred_points))):
                # print(env, i, np.any(np.isnan(np.array(bbox_pred_points))))
                # print(bbox_pred_points)
                # print(model_outputs_all["box_features"][env,batch_inds,:,:])
                bbox_pred_points = torch.zeros_like(bbox_pred_points)
                model_outputs_all["box_features"][env,batch_inds,:,:] = torch.zeros_like(outputs["box_features"].detach().cpu())
            obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
            cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()
            # ipy.embed()
            chair_prob = cls_prob[:,:,3]
            # print(bbox_pred_points, bbox_pred_points.shape)
            # print(obj_prob, obj_prob.shape)
            box_inds = obj_prob.argmax(1) # Indices corresponding to most prominent box for each batch
            # print(box_inds)
            # (box_prob,box_inds) = obj_prob.topk(1, dim=1)
            # print(box_prob, box_inds)
            # sort_box = torch.sort(obj_prob,1,descending=True)
            sort_box = torch.sort(chair_prob,1,descending=True)
            # print('Sorted idx', sort_box[1], 'Array', sort_box[0], box_inds)

            ######
            corners_gt = bboxes_ground_truth[env,:,:,:].numpy()
            # print(corners_gt)

            ######
            # ipy.embed()
            num_chairs_visible = len([i for i, val in enumerate(data[env]["is_visible"][batch_inds][0]) if val])
            g0 = np.zeros((num_chairs_visible,2))
            g1 = np.zeros_like(g0)
            g2 = np.zeros_like(g0)
            chair_idx = 0
            for j, val in enumerate(data[env]["is_visible"][batch_inds][0]):
                if val:
                    g0[chair_idx,:] = [corners_gt[j,0,0], corners_gt[j,1,0]]
                    g1[chair_idx,:] = [corners_gt[j,0,1], corners_gt[j,1,1]]
                    g2[chair_idx,:] = [corners_gt[j,0,2], corners_gt[j,1,2]]
                    chair_idx = chair_idx+1
            #######
            pc_plot = pc[:, pc[0,:,2] > 0.0,:]

            # Visualize
            if visualize:
                plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter3D(
                    pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
                )
            num_probs = 0
            corners = []
            sort_with_nms = []
            for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
            # for (sorted_idx,prob) in zip(list(sort_box[1][:]), list(sort_box[0][:])):
                # print((model_outputs_all["sem_cls_prob"][env,batch_inds,sorted_idx,:])/torch.sum(model_outputs_all["sem_cls_prob"][env,batch_inds,sorted_idx,:]))
                p_thresh = 0.05*obj_prob[0,sorted_idx]
                is_chair = (chair_prob[:,sorted_idx] > p_thresh) # check if chair
                is_object = obj_prob[0,sorted_idx]
                # print("Objectness prob: ", is_object, "Chair prob: ",chair_prob[:,sorted_idx])
                # if(num_probs <num_boxes and is_object>0.1 and is_chair):
                if (num_probs < num_boxes):
                # if(num_probs <num_boxes and is_chair):
                    # num_probs +=1
                    # print("Objectness prob: ", is_object, "Chair prob: ",chair_prob[:,sorted_idx])
                    prob = prob.numpy()
                    bbox = bbox_pred_points[range(batch_size), sorted_idx, :, :]
                    cc = pc_to_axis_aligned_rep(bbox.numpy())
                    # print(pos)
                    dist_from_box = (((pos[0]-cc[0,0,0]/2-cc[0,1,0]/2)**2 + (pos[1]-cc[0,0,1]/2-cc[0,1,1]/2)**2)**0.5)
                    flag = False
                    if num_probs == 0:
                        corners.append(cc)
                        num_probs +=1
                        sort_with_nms.append(sorted_idx)
                    # elif len(sort_with_nms) < num_chairs:
                    else:
                        # print(len(sort_with_nms), sort_with_nms)
                        for cc_keep in corners:
                            bb1 = (cc_keep[0,0,0],cc_keep[0,0,1],cc_keep[0,1,0],cc_keep[0,1,1])
                            bb2 = (cc[0,0,0],cc[0,0,1],cc[0,1,0],cc[0,1,1])
                            # Non-maximal supression, check if IoU more than some threshold to keep box
                            if(box2d_iou(bb1,bb2) > 0.1):
                                flag = True
                        if not flag:    
                            corners.append(cc)
                            num_probs +=1
                            sort_with_nms.append(sorted_idx)
                    # else:
                    #     flag = True
                    # print(sort_box[1][0,0:num_boxes], torch.tensor(sort_with_nms))
                    r0 = [cc[0,0, 0], cc[0,1, 0]]
                    r1 = [cc[0,0, 1], cc[0,1, 1]]
                    r2 = [cc[0,0, 2], cc[0,1, 2]]
                    # ipy.embed()
                    for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                        if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                            if (visualize and not flag):
                                ax.plot3D(*zip(s, e), color=(0.5+0.5*prob, 0.1,0.1))
        
            for j in range(num_chairs_visible):
                for s, e in combinations(np.array(list(product(g0[j], g1[j], g2[j]))), 2):
                    if (np.sum(np.abs(s-e)) == g0[j,1]-g0[j,0] or np.sum(np.abs(s-e)) == g1[j,1]-g1[j,0] or np.sum(np.abs(s-e)) == g2[j,1]-g2[j,0]):
                        if visualize:
                            ax.plot3D(*zip(s, e), color="g")

            if visualize:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()
            
            # Check if object was actually visible; if not, mask loss
            # TODO: Right now, we are just using the output of 3DETR to check this. We need to do this properly (see notes).
            # p_thresh = 0.2
            # is_object = (outputs["outputs"]["objectness_prob"].amax(1) > p_thresh)
            # is_chair = (outputs["outputs"]["sem_cls_prob"][:,:,3].amax(1) > p_thresh) # check if chair
            # is_visible = torch.logical_and(is_object, is_chair)
            is_visible = data[env]["is_visible"][batch_inds][0]
            is_visible = torch.tensor(is_visible).to(device)

            # Check if camera was inside obstacle bounding box (and 0 out loss if so)
            not_inside = data[env]["cam_not_inside_obs"][batch_inds][0]
            not_inside = torch.tensor(not_inside).to(device)
            # print(is_visible, not_inside)
            # loss_mask[env, batch_inds] =  torch.logical_and(is_visible, not_inside).float()
            loss_mask[env, batch_inds, :] =  is_visible.float()
            # print("Loss: ", loss_mask[env, batch_inds])

            # bbox_pred_points = bbox_pred_points[range(batch_size), box_inds, :, :]
            # bbox_pred_points = bbox_pred_points[range(batch_size), sort_box[1][0:num_boxes], :, :]

            # model_outputs_all["box_axis_aligned"][env,batch_inds,:,:] = torch.tensor(pc_to_axis_aligned_rep(bbox_pred_points.numpy()))
            if not np.any(np.isnan(np.array(outputs["outputs"]["box_corners"].detach().cpu()))):
                idx_nms = torch.tensor(sort_with_nms)
                for k in range(len(idx_nms)):
                    box = torch.clone(bbox_pred_points[range(batch_size), idx_nms[k],:,:])
                    most_likely_boxes[env,batch_inds,k,:,:] = torch.tensor(pc_to_axis_aligned_rep(box.numpy()))
            else:
                room_size = 8
                for k in range(1):
                    box = torch.tensor(pc_to_axis_aligned_rep((bbox_pred_points[range(batch_size), sort_box[1][0,k],:,:].numpy())))
                    # box = torch.tensor(pc_to_axis_aligned_rep(bbox_pred_points.numpy()))
                    box[:,0,1] = 0*torch.ones_like(box[:,0,0])
                    box[:,0,0] = (-room_size/2)*torch.ones_like(box[:,0,1])
                    box[:,0,2] = 0*torch.ones_like(box[:,0,2])
                    box[:,1,1] = room_size*torch.ones_like(box[:,1,0])
                    box[:,1,0] = (room_size/2)*torch.ones_like(box[:,1,1])
                    box[:,1,2] = room_size*torch.ones_like(box[:,1,2])
                    most_likely_boxes[env,batch_inds,k,:,:] = box.clone().detach()
                loss_mask[env, batch_inds,:] =  torch.zeros(num_chairs).float()
                # print("No BB found, saving a room sized box")
            #####################################

            # Assign the correct box prediction to each ground truth (visible) box
            max_iou = torch.zeros(num_chairs)
            center_diff = 10*torch.ones(num_chairs)
            sorted_pred = torch.clone(model_outputs_all["box_axis_aligned"][env,batch_inds,:,:,:])
            for j, val in enumerate(data[env]["is_visible"][batch_inds][0]):
                if val:
                    gt = (corners_gt[j,0,0], corners_gt[j,0,1], corners_gt[j,1,0], corners_gt[j,1,1])
                    # print("Ground truth box: ", corners_gt[j,:,:])
                    for kk in range(num_boxes):
                        pred_ = most_likely_boxes[env,batch_inds,kk,:,:]
                        pred = (pred_[0,0,0], pred_[0,0,1], pred_[0,1,0], pred_[0,1,1])
                        iou = box2d_iou(pred, gt)
                        diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                        # print("Prediction: ", pred_)
                        if iou > max_iou[j]:
                            # print("IoU: ", iou, " Max IoU: ", max_iou[j])
                            max_iou[j] = iou
                            sorted_pred[0,j,:,:] = pred_
                            center_diff[j] = diff
                        elif iou == 0 and max_iou[j] == 0 and (center_diff[j] > diff):
                            # Centers of the predicted box are closer than before
                            # print("Center diff: ", diff, " Diff so far: ", center_diff[j])
                            center_diff[j] = diff
                            sorted_pred[0,j,:,:] = pred_
            # print("Before sorting: ", model_outputs_all["box_axis_aligned"][env,batch_inds,:,:,:])
            # print("After sorting: ", sorted_pred)
            # print("Ground truth: ", corners_gt)
            # print("Loss mask", loss_mask[env, batch_inds,:])
            # ipy.embed()
            model_outputs_all["box_axis_aligned"][env,batch_inds,:,:,:] = sorted_pred

    t_end = time.time()
    print("Time: ", t_end - t_start)
    ###########################################################################



    ###########################################################################
    # # Save processed feature data
    torch.save(model_outputs_all, "data/features_multiple_chairs.pt")

    # # Save ground truth bounding boxes
    torch.save(bboxes_ground_truth_aligned, "data/bbox_labels_multiple_chairs.pt")

    # # Save loss mask
    torch.save(loss_mask, "data/loss_mask_multiple_chairs.pt")
    ###########################################################################


