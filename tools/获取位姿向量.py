from elephant import elephant_command
from math import *
import keyboard


# 需要把坐标的单位进行转化，matlab程序中要求长度单位是m，角度单位是弧度
# elephant默认是mm以及角度,保留小数点后6位
def trans_unit_hudu(cur_pose):
    cur_pose[0] = cur_pose[0]/1000
    cur_pose[1] = cur_pose[1]/1000
    cur_pose[2] = cur_pose[2]/1000
    cur_pose[3] = cur_pose[3]/180*pi
    cur_pose[4] = cur_pose[4]/180*pi
    cur_pose[5]= cur_pose[5]/180*pi
    for j in range(0,6):
        cur_pose[j]=round(cur_pose[j],6)
    return cur_pose


def trans_unit_jiaodu(cur_pose):
    cur_pose[0] = cur_pose[0]/1000
    cur_pose[1] = cur_pose[1]/1000
    cur_pose[2] = cur_pose[2]/1000
    for j in range(0,6):
        cur_pose[j]=round(cur_pose[j],6)
    return cur_pose

# 创建对象
erobot = elephant_command()


print("按q开始录入\n")

# 存放于一个CSV


# # 存放于单独的txt
# count=0
# shiwei=0
# while True:
#     keyboard.wait('q')
#     cur_pose = erobot.get_coords()
#     cur_pose = trans_unit_jiaodu(cur_pose)
#     print("转换单位后的笛卡尔坐标系)：\n", cur_pose)
#     with open('D:/company/py/yolo/calibration_file/poses{}{}.txt'.format(shiwei, count), 'w') as f:
#         for i in range(0,6):
#             f.write(str(cur_pose[i]))
#             if i!=5:
#                 f.write(' ')
#     if count==9:
#         shiwei+=1
#         count=0
#         continue
#     count += 1



