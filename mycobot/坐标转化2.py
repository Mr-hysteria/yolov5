"""
已经获取了末端坐标，将其转化成base坐标
1.连接机械手，获取末端位置
2.根据末端欧拉角获取转化矩阵
3.把末端位置转化成base坐标系下
"""
import numpy as np
from elephant import elephant_command
from scipy.spatial.transform import Rotation as R
from math import *


def combine_R_T(R, T):
    rt = np.column_stack([R, T])  # 列合并
    rt = np.row_stack((rt, np.array([0, 0, 0, 1])))
    rt = np.matrix(rt)  # array转化成matrix，方便求逆矩阵
    return rt


# 末端位置根据前面的代码给出
object_end_pose = np.array([[1, 2, 3, 1]])

# 通过socket获取机器臂末端欧拉角(x,y,z,Rx,Ry,Rz)
erobot = elephant_command()
end_pose = erobot.get_coords()
print("目前坐标位置：\n", end_pose)
# 得到base to end的转化矩阵
end = [end_pose[5], end_pose[4], end_pose[3]]  # zyx顺序
ret = R.from_euler('ZYX', end, degrees=True)
r = ret.as_matrix()
t = [end_pose[0], end_pose[1], end_pose[2]]
RT_end_base = combine_R_T(r,t)
# 末端位置在base中的坐标为：
final = RT_end_base.I @ object_end_pose.T

