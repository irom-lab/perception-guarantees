import pyzed.sl as sl
import argparse
from scipy.spatial.transform import Rotation
import tf
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
sys.path.append('datasets')

import torch
import torch.nn as nn

from itertools import product, combinations
from models import build_model
from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
dataset_config = dataset_config()
# Mesh IO
import numpy as np
from utils.make_args import make_args_parser
from utils.pc_util import preprocess_point_cloud, pc_to_axis_aligned_rep
from utils.box_util import box2d_iou

class Zed:
    def __init__(self,state_ic=[0.0,0.0,0.0,0.0], yaw_ic=0.0):
        # init camera
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
        parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
        parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
        self.opt = parser.parse_args()
        if len(self.opt.input_svo_file)>0 and len(self.opt.ip_address)>0:
            print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
            exit()
        init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL,
                                coordinate_units=sl.UNIT.METER,
                                coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
                                depth_minimum_distance = 1,
                                depth_maximum_distance = 5.5)
        self.parse_args(init)
    
        
        parser = make_args_parser()
        args = parser.parse_args(args=[])
        # Dataset config: use SUNRGB-D
        # dataset_config = dataset_config()
        # Build model
        self.model, _ = build_model(args, dataset_config)

        # Load pre-trained weights
        sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
        self.model.load_state_dict(sd["model"]) 

        self.model = self.model.cuda()
        self.model.eval()

        self.device = torch.device("cuda")

        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        res = sl.Resolution()
        res.width = 720
        res.height = 404

        initial_position = sl.Transform()
        # Set the initial position of the Camera Frame at vicon initial  above the World Frame
        initial_translation = sl.Translation()
        initial_rotation = sl.Rotation()
        initial_translation.init_vector(state_ic[0], state_ic[1], 0.42)
        initial_rotation.set_euler_angles(0,0,yaw_ic)
        initial_position.set_translation(initial_translation)
        initial_position.set_euler_angles(0,0,yaw_ic)

        tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
        tracking_params.set_initial_world_transform(initial_position)
        tracking_params.enable_imu_fusion = True
        tracking_params.mode = 2
        status = self.zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
        if status != sl.ERROR_CODE.SUCCESS:
            print("Enable Positional Tracking : "+repr(status)+". Exit program.")
            self.zed.close()
            exit()

        self.runtime = sl.RuntimeParameters()
        self.runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
        self.camera_pose = sl.Pose()
        self.py_translation = sl.Translation()
        self.py_orientation = sl.Orientation()
        # camera + 3DETR
        self.num_pc_points = 40000
        self.pc = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    def get_IMU(self):
        ts_handler = TimestampHandler()
        sensors_data = sl.SensorsData()
        
        if self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS :
            # Check if IMU data has been updated
            if ts_handler.is_new(sensors_data.get_imu_data()):
                quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                acceleration = sensors_data.get_imu_data().get_linear_acceleration()
        
        return quaternion, acceleration
    
    def get_pose(self):
        t_translation = 0.0
        yaw = 0.0
        is_success = (self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS)
        while not is_success:
            is_success =self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS
            print("DEBUG: No success in grabbing :(")
        tracking_state = self.zed.get_position(self.camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
        is_tracking = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
        # while not is_tracking:
        #     is_tracking = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
        #     print("DEBUG: No success in tracking :(")
        #Get rotation and translation and displays it
        rotation = self.camera_pose.get_rotation_vector()
        translation = self.camera_pose.get_translation(self.py_translation)
        t_translation = [round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)]
        pose_data = self.camera_pose.pose_data(sl.Transform())
        # print(t_translation, self.camera_pose.timestamp.get_microseconds())

        #Orientation quaternion
        ox = round(self.camera_pose.get_orientation(self.py_orientation).get()[0], 3)
        oy = round(self.camera_pose.get_orientation(self.py_orientation).get()[1], 3)
        oz = round(self.camera_pose.get_orientation(self.py_orientation).get()[2], 3)
        ow = round(self.camera_pose.get_orientation(self.py_orientation).get()[3], 3)
        q = [ox, oy, oz, ow]
        
        yaw = self.get_yaw_from_quat(q)
                # euler = tf.transformations.euler_from_quaternion(q)
                # yaw =  euler[2] # TODO: FIX this. Yaw is definitely wrong
            # else:
                # self.get_pose()
        # else:
            # self.get_pose()
        return t_translation, self.camera_pose.timestamp.get_microseconds(), yaw

    def get_yaw_from_quat(self, quat):
        # TODO: update this. yaw is wrong
        rot = Rotation.from_quat(quat)
        yaw = rot.as_euler('xyz', degrees=False)[2]

        return yaw
    
    def get_boxes(self, cp=0.4, num_boxes=10):
        # TODO: add zed related functions for using the pointcloud
        # Get pointcloud
        # self.zed.retrieve_measure(self.pc, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)

        # pc = sl.Mat()
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.pc,  sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            # Get pointcloud as a numpy array (remove nan, -inf, +inf and RGB values)
            points = np.array(self.pc.get_data())
            #print(np.shape(points), np.min(points), np.max(points))
            points = points[:,:,0:3]
            points = points[np.isfinite(points).any(axis=2),: ]
            points = points[points[:,2]<2,:]
            # points_temp = np.copy(points)
            # points[:,2] = points_temp[:,1]
            # points[:,0] = -points_temp[:,2]
            # points[:,1] = -points_temp[:,0]
            # print(np.shape(points), np.min(points), np.max(points))
            if (len(points[0])>0):
                batch_size = 1
                points = np.array(points).astype('float32')
                # Downsample
                points_ds = preprocess_point_cloud(points, self.num_pc_points)
                points_ds = points_ds.reshape((batch_size, self.num_pc_points, 3))
                pc_all = torch.from_numpy(points_ds).to(self.device)
                boxes, box_features = self.get_box_from_pc(pc_all, cp, num_boxes, False)
            else:
                points = np.zeros((1,self.num_pc_points, 3),dtype='float32')
                pc_all = torch.from_numpy(points).to(self.device)
                boxes, box_features = self.get_room_size_box(pc_all)
        else:
            points = np.zeros((1,self.num_pc_points, 3),dtype='float32')
            pc_all = torch.from_numpy(points).to(self.device)
            boxes, box_features = self.get_room_size_box(pc_all)
        return boxes

    def get_box_from_pc(self, pc_all, cp, num_boxes, visualize=False):
        pc_min_all = pc_all.min(1).values
        pc_max_all = pc_all.max(1).values
        inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

        outputs = self.model(inputs)
        bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
        # print(bbox_pred_points.shape)
        cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()

        # model_outputs_all["box_corners"][env,batch_inds,:,:] = outputs["outputs"]["box_corners"].detach().cpu()
        # model_outputs_all["box_features"][env,batch_inds,:,:] = outputs["box_features"].detach().cpu()

        box_features = outputs["box_features"].detach().cpu()

        chair_prob = cls_prob[:,:,3]
        obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
        sort_box = torch.sort(obj_prob,1,descending=True)

        num_probs = 0
        boxes = np.zeros((num_boxes, 2,3))
        if np.any(np.isnan(np.array(bbox_pred_points))):
            return self.get_room_size_box(pc_all)
        
        for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
            if (num_probs < num_boxes):
                prob = prob.numpy()
                bbox = bbox_pred_points[0, sorted_idx, :, :]
                bb = pc_to_axis_aligned_rep(bbox.numpy())
                # print(cc)
                flag = False
                if num_probs == 0:
                    boxes[num_probs,:,:] = bb
                    num_probs +=1
                else:
                    for bb_keep in boxes:
                        bb1 = (bb_keep[0,0],bb_keep[0,1],bb_keep[1,0],bb_keep[1,1])
                        bb2 = (bb[0,0],bb[0,1],bb[1,0],bb[1,1])
                        # Non-maximal supression, check if IoU more than some threshold to keep box
                        if(box2d_iou(bb1,bb2) > 0.1):
                            flag = True
                    if not flag:    
                        boxes[num_probs,:,:] = bb
                        num_probs +=1
        boxes[:,0,:] -= cp
        boxes[:,1,:] += cp
        return boxes, box_features

    def get_room_size_box(self, pc_all):
        room_size = 8
        num_chairs =5
        boxes = np.zeros((num_chairs, 2,3))
        # box = torch.tensor(pc_to_axis_aligned_rep(bbox_pred_points.numpy()))
        boxes[:,0,1] = 0*np.ones_like(boxes[:,0,0])
        boxes[:,0,0] = (-room_size/2)*np.ones_like(boxes[:,0,1])
        boxes[:,0,2] = 0*np.ones_like(boxes[:,0,2])
        boxes[:,1,1] = room_size*np.ones_like(boxes[:,1,0])
        boxes[:,1,0] = (room_size/2)*np.ones_like(boxes[:,1,1])
        boxes[:,1,2] = room_size*np.ones_like(boxes[:,1,2])

        pc_min_all = pc_all.min(1).values
        pc_max_all = pc_all.max(1).values
        inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

        outputs = self.model(inputs)
        box_features = outputs["box_features"].detach().cpu()
        return boxes, box_features

    def parse_args(self, init):
        if len(self.opt.input_svo_file)>0 and self.opt.input_svo_file.endswith(".svo"):
            init.set_from_svo_file(self.opt.input_svo_file)
            print("[Sample] Using SVO File input: {0}".format(self.opt.input_svo_file))
        elif len(self.opt.ip_address)>0 :
            ip_str = self.opt.ip_address
            if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
                init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
                print("[Sample] Using Stream input, IP : ",ip_str)
            elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
                init.set_from_stream(ip_str)
                print("[Sample] Using Stream input, IP : ",ip_str)
            else :
                print("Unvalid IP format. Using live stream")
        if ("HD2K" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.HD2K
            print("[Sample] Using Camera in resolution HD2K")
        elif ("HD1200" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.HD1200
            print("[Sample] Using Camera in resolution HD1200")
        elif ("HD1080" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.HD1080
            print("[Sample] Using Camera in resolution HD1080")
        elif ("HD720" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.HD720
            print("[Sample] Using Camera in resolution HD720")
        elif ("SVGA" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.SVGA
            print("[Sample] Using Camera in resolution SVGA")
        elif ("VGA" in self.opt.resolution):
            init.camera_resolution = sl.RESOLUTION.VGA
            print("[Sample] Using Camera in resolution VGA")
        elif len(self.opt.resolution)>0: 
            print("[Sample] No valid resolution entered. Using default")
        else : 
            print("[Sample] Using default resolution") 


class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
        self.t_baro = sl.Timestamp()
        self.t_mag = sl.Timestamp()

    ##
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.MagnetometerData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_mag.get_microseconds())
            if new_:
                self.t_mag = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.BarometerData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_baro.get_microseconds())
            if new_:
                self.t_baro = sensor.timestamp
            return new_

    # def odom_callback(self, msg):
    #     self.x = msg.pose.pose.position.x
    #     self.y = msg.pose.pose.position.y
    #     self.z = msg.pose.pose.position.z

    #     self.vx = 0 # update
    #     self.vy = 0

    #     self.quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    #     self.yaw = self.get_yaw_from_quat(self.quat)

    # def get_yaw_from_quat(self, quat):
    #     rot = Rotation.from_quat(quat)
    #     yaw = rot.as_euler('xyz', degrees=True)[2]

    #     return yaw
