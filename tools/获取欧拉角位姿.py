from elephant import elephant_command
from math import *
import numpy as np
# import keyboard

# 用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


# 用于根据位姿计算变换矩阵(end to base)
# def pose_robot(cur_pose):
#     x = cur_pose[0]; y = cur_pose[1]; z = cur_pose[2]
#     Tx = cur_pose[3]; Ty = cur_pose[4]; Tz = cur_pose[5]
#     thetaX = Tx / 180 * pi
#     thetaY = Ty / 180 * pi
#     thetaZ = Tz / 180 * pi
#     R = myRPY2R_robot(thetaX, thetaY, thetaZ)
#     t = np.array([[x], [y], [z]])
#     RT1 = np.column_stack([R, t])  # 列合并
#     RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
#     return RT1





# erobot = elephant_command()

# # 获取笛卡尔位姿
# cur_pose = erobot.get_coords()
# print("末端的笛卡尔坐标系为：", cur_pose)
# POSE = pose_robot(cur_pose)
# print("笛卡尔坐标系转化成位姿矩阵为：", POSE)

# 获取角度位姿
# cur_angles = erobot.get_angles()
# print(cur_angles)

# cur_pose = erobot.get_coords()
# print(cur_pose)

# count=0;shiwei=0
# print("按q开始录入\n")
# while True:
#     keyboard.wait('q')
#     cur_pose = erobot.get_coords()
#     POSE = pose_robot(cur_pose)
#     print("笛卡尔坐标系转化成位姿矩阵为：\n", POSE)
#     with open('D:/company/py/yolo/calibration_file/poses{}{}.txt'.format(shiwei, count), 'w') as f:
#         for i in range(0,4):
#             for j in range(0,4):
#                 f.write(str(POSE[i][j]))
#                 if j!=3:
#                     f.write(' ')
#                 else:
#                     f.write('\n')
#     if count==9:
#         shiwei+=1
#         count=0
#         continue
#     count += 1
#


# vector = [[ 1.00000000e+00 -2.74452529e-06  2.55936075e-06]
#  [-2.65358765e-06 -3.49007933e-02  9.99390782e-01]
#  [-2.65352955e-06 -9.99390782e-01 -3.49007933e-02]]
R = myRPY2R_robot(-105.424/180*pi, 3.43889/180*pi, 88.7023/180*pi)
print(R)