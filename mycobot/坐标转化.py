"""
把相机坐标转化到end坐标，
标定打坐标系可能版本不同，由于自己用的是rviz中的坐标系，和实际getcoords返回的还不同，但是xyz一样，因此还需要进行一次变换

"""
from math import *
import numpy as np
from scipy.spatial.transform import Rotation as R


# 初始化相机标定结果,注意四元数顺序
euler = [-0.0456701, -0.0247697, 0.11437, -105.424, 3.43889, 88.7023]  # 手眼标定结果end to cam，RPY角
# quat=[-0.58136017537009, -0.542918236178846, 0.44038207047493766, 0.41632171132708873]  # 四元数（xyzw顺序）

# 相机坐标系下的点(增加1个维度),两层[],后续给出
object_cam_pose = np.array([[1, 2, 3, 1]])


# def rviz_to_robot(r):




# 转化为位移矩阵
def trans_matrix(euler):
    x = euler[0]; y = euler[1]; z = euler[2]
    r = [euler[3], euler[4], euler[5]]
    # 欧拉角转旋转矩阵
    r4 = R.from_euler('xyz', r, degrees=True)
    r4 = r4.as_matrix()
    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([r4, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    RT1 = np.matrix(RT1)  # array转化成matrix，方便求逆矩阵
    return RT1


# 获取end to cam!!!!!!!!!!!!!!!!!
RT = trans_matrix(euler)


# 通过.I求逆，获取cam to end
RT_T = RT.I
# 把相机坐标转化成末端坐标系下的坐标
object_end_pos = RT_T @ object_cam_pose.T


# 获取末端位置，获取转换矩阵，吧末端坐标系变成base坐标系（接下来的工作）




