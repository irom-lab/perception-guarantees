from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time


state_type = 'vicon'
vicon = True
go1 = Go1_move([1, 0], vicon=vicon, state_type=state_type)
go1.get_state()

st = time.time()
go1.move([0.11, 0])
# go1.move([0, 0.5])
time.sleep(2)
et = time.time()
print(st - et)