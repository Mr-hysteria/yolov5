# -*- coding: utf-8 -*-

from elephant import elephant_command
import time
import random

erobot = elephant_command()
# 笛卡尔空间
# (x,y,z,RX,Ry,Rz)
cur_pose = erobot.get_coords()
print(cur_pose)
# coords= [-0.048631, 183.859911, 808.44005, 89.999865, -0.263747, 179.911615]
# erobot.set_coords(coords, 100)


# X0 = 100.0
# Y0 = 360.0
# Z0 = 375.0 # mm
# RX0 = 180.0
# RY0 = 0.0
# RZ0 = -22.0
# ZG = 180.0 # mm

# cur_poses_str = erobot.get_coords()
# print(cur_poses_str)
# cur_poses = [float(p) for p in cur_poses_str[12:-1].split(',')]
# print(cur_poses)
# cur_poses[5] = -10

# erobot.set_coords(cur_poses, 500)
# time.sleep(2)
# cur_poses_str = erobot.get_coords()
# print(cur_poses_str)
# # erobot.set_coord('rz', -90, 800)
# # erobot.set_coord('z', 360, 800)
# time.sleep(2)

# 9.14测试

# cur_pose[2] = 200
# erobot.set_coords(cur_pose, 800)
# time.sleep(0.1)
# while erobot.check_running():
#     print('running', time.time())
#     time.sleep(0.1)

# erobot.set_coord('z', 250, 800)
# time.sleep(0.1)
# while erobot.check_running():
#     print('running')
#     time.sleep(0.1)

# 关节空间
'''
 cur_angles = erobot.get_angles()
 print(cur_angles)
 cur_angles[0] = -80.0
 erobot.set_angles(cur_angles, 500)
 time.sleep(0.1)
 run_flag = erobot.check_running()
 while run_flag:
     print('running')
     time.sleep(0.1)
     run_flag = erobot.check_running()

 erobot.set_angle(1, -40, 500)
 time.sleep(0.1)
 while erobot.check_running():
     print('running')
     time.sleep(0.1)
'''
#


# # 速度控制
# erobot.jog_coord('x',1,1000)
# erobot.jog_coord('y',1,1000)
# erobot.jog_coord('rz',1,300)
# # erobot.jog_angle(6,1,500)
# time.sleep(2)
# erobot.jog_coord('x',-1,1000)
# erobot.jog_coord('y',-1,1000)
# erobot.jog_coord('rz',-1,300)
# # erobot.jog_angle(6,-1,500)
# time.sleep(2)
# erobot.jog_coord('x',0,200)
# erobot.jog_coord('y',0,200)
# erobot.jog_coord('rz',0,200)
# # erobot.jog_angle(6,0,500)
# time.sleep(0.2)

# erobot.jog_stop('x')
# erobot.jog_stop('y')
# erobot.jog_stop('rz')

# # 笛卡尔空间
# cur_pose = erobot.get_coords()
# print(cur_pose)
# cur_pose[2] = 200
# erobot.set_coords(cur_pose, 800)
# time.sleep(0.1)
#


# I/O
# time.sleep(0.5)
# for _ in range(5):
#     erobot.set_digital_out(0,1)
#     time.sleep(1)
#     erobot.set_digital_out(0,0)
#     time.sleep(1)
# erobot.set_digital_out(0, 0)
# time.sleep(0.5)

# print(erobot.check_running())
# time.sleep(0.1)
