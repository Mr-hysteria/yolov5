"""
可能需要修改的地方：
1.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)，修改帧率分辨率（此处是bgr，目的是用opencv的imshow与imwrite正常展示）
2.储存照片的路径(pycharm直接复制绝对路径即可)

使用方法：
1.空格采集（最好不要太模糊的图）
2.回车自动采集（最好不用）
3.ESC退出
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Configure depth and color streams
pipeline = rs.pipeline()
# 创建 config 对象：
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 训练得用rgb来做,opencv显示则需要用bgr，不然颜色就有问题
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        # color_image_trans = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        c = cv2.waitKey(1)

        # 手动采集（空格键）
        # 敲击空格将图像保存(空格的ascii码是32)
        if c == 32:

            # 计算文件夹里jpg文件数量，以便于关闭软件后重新打开采集不会将已有图片覆盖
            count = 0
            for filename in os.listdir('/home/desktop-tjj/yolov5/tools/capture_photos'):
                if filename.endswith('.jpg'):
                    count += 1
            # print(count)
            # 这是opencv的写入，需要写入bgr图像才会正常的显示
            cv2.imwrite('/home/desktop-tjj/yolov5/tools/capture_photos/{}.jpg'.format(count + 1), color_image)

        # 自动采集（回车键）
        # 如果按下回车键则自动采集（回车键ascii码是13）
        if c == 1333:
            # 获取时间以在循环中判断是否经过了某段时间
            time0 = time.time()
            while True:
                # 之前卡死是因为进了这个循环出不去了了，没法wait_for_frames同时opencv窗口也没法waitKey刷新所以就卡死了，所以必须再把wait_for_frames和waitKey加进来

                # Wait for a coherent pair of frames（一对连贯的帧）: depth and color
                frames = pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

                cv2.imshow('RealSense', color_image)

                c = cv2.waitKey(1)

                # 计算文件夹里jpg文件数量，以便于关闭软件后重新采集不会将已有图片覆盖
                count = 0
                for filename in os.listdir('/home/desktop-tjj/yolov5/tools/capture_photos'):
                    if filename.endswith('.jpg'):
                        count += 1

                # 设置延时时间（单位：秒）
                time_delay = 1

                # 使用经过的时间差来对操作进行延时，这样可避免流停滞，但此方法不完全准确，特别是在我们的操作耗时较长的情况下，延时误差较大。当我们将操作的时间视为无穷小时，此方法准确
                if (time.time() - time0) > time_delay:
                    # 可以看到，实际延时微微大于我们设置的延时时间
                    print(time.time() - time0)
                    # 1.0003252029418945
                    # 1.0017242431640625
                    # 1.0030531883239746
                    # 1.004326343536377
                    # 1.0044221878051758
                    # 1.008774995803833
                    # 1.005782127380371
                    # 1.0061774253845215
                    # 1.010535717010498

                    # 保存图像，保存到上一层的imgs文件夹内，以1、2、3、4...为文件名保存图像
                    cv2.imwrite('/home/desktop-tjj/yolov5/tools/capture_photos/{}.jpg'.format(count + 1), color_image)

                    # 只让操作在某时间段内只执行一次，下次执行需在延时时间time_delay（秒）后
                    time0 += time_delay

                # 延时一下，不然保存太快翻车咋办
                # 之前卡顿的原因是用了time.sleep函数，导致wait_for_frames变慢了，不能用sleep函数
                # time.sleep(1)

                if c == 27:
                    break

        # 如果按下ESC则关闭窗口（ESC的ascii码为27），同时跳出循环
        if c == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()

