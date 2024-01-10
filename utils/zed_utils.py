import pyzed.sl as sl
import argparse
from scipy.spatial.transform import Rotation
import tf

class Zed:
    def __init__(self):
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
                                depth_minimum_distance = 0.15,
                                depth_maximum_distance = 8)
        self.parse_args(init)

        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        res = sl.Resolution()
        res.width = 720
        res.height = 404

        tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
        tracking_params.enable_imu_fusion = True
        tracking_params.mode = 2
        status = self.zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
        if status != sl.ERROR_CODE.SUCCESS:
            print("Enable Positional Tracking : "+repr(status)+". Exit program.")
            self.zed.close()
            exit()

        self.runtime = sl.RuntimeParameters()
        self.camera_pose = sl.Pose()
        self.py_translation = sl.Translation()
        self.py_orientation = sl.Orientation()

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
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = self.zed.get_position(self.camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
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

        return t_translation, self.camera_pose.timestamp.get_microseconds(), yaw

    def get_yaw_from_quat(self, quat):
        # TODO: update this. yaw is wrong
        rot = Rotation.from_quat(quat)
        yaw = rot.as_euler('xyz', degrees=False)[2]

        return yaw
    
    def get_pc(self):
        # TODO: add zed related functions for using the pointcloud
        pass

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
