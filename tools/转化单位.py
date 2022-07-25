from math import *

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
        # 更新数据,以下可以更新为°，根据需要取舍
        # cur_pose[3] = cur_pose[3]*180/pi
        # cur_pose[4] = cur_pose[4]*180/pi
        # cur_pose[5] = cur_pose[5]*180/pi
        # 写入新的文件中
        with open('D:/company/py/yolo/calibration_file/trans/poses.txt', 'a') as F:
            # F.write('hand')
            # F.write(',')
            for i in range(0,6):
                F.write(str(round(cur_pose[i],6)))
                if i!=5:
                    F.write(',')
                if i==5:
                    F.write('\n')
    if count==9 and shiwei==1:
        break
    if count==9:
        shiwei+=1
        count=0
        continue
    count += 1
