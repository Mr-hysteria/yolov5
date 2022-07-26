import pyrealsense2 as rs
import cv2
import random
import torch
import time
from scipy.spatial.transform import Rotation as R
from elephant import elephant_command
from global_v import *
import threading

# 初始化重要全局变量
camera_coordinate_4d = [0, 0, 0, 1]  # 目标点在相机坐标系下的描述，单位是米
base_coordinate_4d = [0, 0, 0, 1]  # 齐次式
euler = euler_g
end_to_rviz = end_to_rviz_g
initP = initP_g
initA = initA_g
erobot = elephant_command()
key = 0


# 进行平面移动,偏移量逐渐接近测得值，因为开始会误差大
def move_1():
    global camera_coordinate_4d
    global erobot
    while True:
        robot_pos = erobot.get_coords()
        A = camera_coordinate_4d
        # position = [robot_pos[0], robot_pos[1] + A[0], robot_pos[2] - A[1], initP[3], initP[4], initP[5]]  # 桌子用这个
        position = [robot_pos[0] + A[0], robot_pos[1] , robot_pos[2]- A[1], initP[3], initP[4], initP[5]]  # 小车用这个
        # print('末端需要移动到的位置：\n', position)
        erobot.set_coords(position, 2000)
        time.sleep(3)  # 此处不能用wait_command_done()，不然后台会被冻结
        print("如果调整好则按ESC或者q")
        global key
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


def move_2():
    global base_coordinate_4d
    global erobot
    A = base_coordinate_4d
    position = [A[0], A[1] - 110, A[2], initP[3], initP[4], initP[5]]
    erobot.set_coords(position, 2000)
    time.sleep(8)  # 双重保险，有时候下面那一句失效
    erobot.wait_command_done()
    time.sleep(2)


def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定中心点
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 以中心点为中心，确定深度搜索范围（方框宽度最小值）
    # randnum是为了多取一些值来取平均
    for i in range(randnum):
        bias = random.randint(-min_val // 4, min_val // 4)  # 随机偏差,控制被除数大小即可控制范围
        dist = depth_data.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))  # 单位为m
        # dist = depth_data[int(mid_pos[0] + bias), int(mid_pos[1] + bias)]

        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
    return np.mean(distance_list)


# 这个函数主要是在原图上画框,标出深度信息,此外用毫米看比较方便
def dectshow(org_img, boxs, depth_data, intrin):
    img = org_img.copy()
    for box in boxs:
        dist = get_mid_pos(org_img, box, depth_data, 24)  # dist单位m，后续统一表示为mm

        camera_coordinate_3d = rs.rs2_deproject_pixel_to_point(intrin=intrin,
                                                               pixel=[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                                                               depth=dist)  # 单位为m
        global camera_coordinate_4d  # 函数内使用全局变量需要声明
        camera_coordinate_4d = np.array(
            [camera_coordinate_3d[0] * 1000, camera_coordinate_3d[1] * 1000, camera_coordinate_3d[2] * 1000, 1])
        global base_coordinate_4d
        base_coordinate_4d1 = trans_base_to_cam(camera_coordinate_4d)
        base_coordinate_4d1 = np.array(base_coordinate_4d1)  # 需要array对象。有两层括号，因此需要提取[0]
        base_coordinate_4d = base_coordinate_4d1[0]

        # print('像素坐标系：', [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, dist * 1000])
        # print('相机坐标系：',
        #       [camera_coordinate_3d[0] * 1000, camera_coordinate_3d[1] * 1000, camera_coordinate_3d[2] * 1000])
        # global euler
        # print("rviz坐标系：", rviz_to_cam_f(euler) @ camera_coordinate_4d)
        # global end_to_rviz
        # print("end坐标系：", end_to_rviz @ rviz_to_cam_f(euler) @ camera_coordinate_4d)
        # global erobot
        # print("机器人末端位置：", erobot.get_coords())
        print("base_coordinate_4d:", base_coordinate_4d)
        print("----------------------------------\n")

        text_pixel = str((int(box[0]) + int(box[2])) // 2) + ', ' + str((int(box[1]) + int(box[3])) // 2) + '(pixel)'
        text_camera = str(camera_coordinate_3d[0] * 1000)[:4] + ', ' + str(camera_coordinate_3d[1] * 1000)[:4] + '(mm)'
        # text_base = str(base_coordinate_4d[0])[:4] + ',' + str(base_coordinate_4d[1])[:4] + ',' + str(dist * 1000)[
        #                                                                                           :4] + '(mm)'

        # rectangle画框，参数表示依次为：(图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)  ###?
        # circle画圆心，参数表示依次为：(img, center, radius, color[, thickness]),thickness为负表示绘制实心圆
        center = [(int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2]
        cv2.circle(img, center, 8, (0, 255, 0), -1)

        # putText各参数依次是：图片，添加的文字(标签+深度-单位m)，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(img, 'cup_distance:' + str(dist * 1000)[:4] + 'mm',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'cup_position_in_pixel:' + text_pixel,
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'cup_position_in_camera:' + text_camera,
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(img, 'cup_position_in_base:' + text_base, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('dec_img', img)


def realsense_detect():  # 进行目标识别，显示目标识别效果，返回相机坐标系下的值
    # 配置yolov5
    model = torch.hub.load('/home/desktop-tjj/yolov5', 'custom', path='/home/desktop-tjj/yolov5/mymodels/black_cup.pt',
                           source='local')  # 加载模型
    model.eval()
    # 配置摄像机参数，用opencv的时候，颜色通道是bgr
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 深度通道的分辨率最大为1280x720
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # 开始采集
    pipeline.start(config)
    # 深度与彩色图像对齐
    alignIt = rs.align(rs.stream.color)
    num = 1

    try:
        while True:
            # 获取深度图以及彩色图像
            frames = pipeline.wait_for_frames()

            # 获取相机内参
            if num == 1:
                color_frame = frames.get_color_frame()
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                num += 1

            aligned_frame = alignIt.process(frames)  # 获取对齐数据
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转化成numpy格式
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 开始预测，转化通道(BGR TO RGB)
            convert_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = model(convert_img)
            boxs = results.pandas().xyxy[0].values
            dectshow(color_image, boxs, depth_frame, intr)  # 用的是depth_frame(没有转化成np格式的)，因为要调用get_distance

            # # 可选，展示彩色图像和深度图
            # # 在深度图像上应用colormap(图像必须先转换为每像素8位)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # # 水平堆叠深度图和彩色图
            # images = np.hstack((color_image, depth_colormap))
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            global key
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def rviz_to_cam_f(euler):  # 把欧拉角转化成4x4变换矩阵(rviz to cam)
    x = euler[0] * 1000
    y = euler[1] * 1000
    z = euler[2] * 1000
    r = [euler[3], euler[4], euler[5]]
    # 欧拉角转旋转矩阵
    r4 = R.from_euler('xyz', r, degrees=True)
    r4 = r4.as_matrix()
    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([r4, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    RT1 = np.matrix(RT1)  # array转化成matrix，方便求逆矩阵
    return RT1


def trans_base_to_end():
    global erobot
    end_pose = erobot.get_coords()
    # print("目前机械臂坐标位置(mm)：\n", end_pose)
    # 得到base to end的转化矩阵
    end = [end_pose[3], end_pose[4], end_pose[5]]
    ret = R.from_euler('xyz', end, degrees=True)
    r = ret.as_matrix()
    t = [end_pose[0], end_pose[1], end_pose[2]]
    rt = np.column_stack([r, t])  # 列合并
    rt = np.row_stack((rt, np.array([0, 0, 0, 1])))
    rt = np.matrix(rt)  # array转化成matrix，方便求逆矩阵
    return rt


def trans_base_to_cam(camera_coordinate):  # 把相机坐标系下的坐标转化为base坐标系下的坐标，根据链式法则求
    global euler
    rviz_to_cam = rviz_to_cam_f(euler)  # 得到矩阵1
    global end_to_rviz
    end_to_rviz1 = np.matrix(end_to_rviz)
    base_to_end = trans_base_to_end()
    return base_to_end @ end_to_rviz1 @ rviz_to_cam @ camera_coordinate
