from scipy.spatial.transform import Rotation as R
import numpy as np
#
# end_pose = [1,2,3,45,45,45]
#
# end = [end_pose[3], end_pose[4], end_pose[5]]
# ret = R.from_euler('xyz', end, degrees=True)
# r = ret.as_matrix()
# print(r)
# t = [end_pose[0], end_pose[1], end_pose[2]]
# rt = np.column_stack([r, t])  # 列合并
# print(rt)
# rt = np.row_stack((rt, np.array([0, 0, 0, 1])))
# rt = np.matrix(rt)  # array转化成matrix，方便求逆矩阵
# print(rt)

# a = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0],
#                            [0, 0, 0, 1]])
# print(a)
