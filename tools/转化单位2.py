from math import *
import numpy as np


def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

count=0
shiwei=0
while True:
    cur_pose = [0] * 6
    with open('D:/company/py/yolo/calibration_file/source/poses{}{}.txt'.format(shiwei, count), 'r') as f:
        # 遍历每一行
        j=0
        for line in f:
            # 对每一行字符串进行分割
            numberlist = line.split()
            for number in numberlist:
                cur_pose[j] = float(number)
                j+=1
        # 更新数据，以下可以更新为mm，根据需要取舍
        cur_pose[0] = cur_pose[0]*1000
        cur_pose[1] = cur_pose[1]*1000
        cur_pose[2] = cur_pose[2]*1000
        # 更新数据,以下可以更新为°，根据需要取舍
        # cur_pose[3] = cur_pose[3]*180/pi
        # cur_pose[4] = cur_pose[4]*180/pi
        # cur_pose[5] = cur_pose[5]*180/pi
        RT = myRPY2R_robot(cur_pose[3],cur_pose[4],cur_pose[5])
        # 写入新的文件中，可以全文本
        with open('D:/company/py/yolo/calibration_file/trans/poses.txt', 'a') as F:
             F.write('Rmarker2world(:,:,{}) = '.format(10*shiwei+(count+1)))
             F.write(str(RT))
             F.write(';\n')
             #    if i!=5:
             #        F.write(',')
             #    if i==5:
             #        F.write('\n')
        with open('D:/company/py/yolo/calibration_file/trans/vector.txt', 'a') as F:
             F.write('Tmarker2world(:,{}) = '.format(10*shiwei+(count+1)))
             F.write('['+str(cur_pose[0])+' '+str(cur_pose[1])+' '+str(cur_pose[2])+']\';')
             F.write('\n')
             #    if i!=5:
             #        F.write(',')
             #    if i==5:
             #        F.write('\n')
    if count==9 and shiwei==1:
        break
    if count==9:
        shiwei+=1
        count=0
        continue
    count += 1
