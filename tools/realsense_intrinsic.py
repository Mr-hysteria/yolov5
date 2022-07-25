import pyrealsense2 as rs
import json
# 相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

pipeline.start(config)
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
# 获取相机内参
intr = color_frame.profile.as_video_stream_profile().intrinsics
camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                     'ppx': intr.ppx, 'ppy': intr.ppy,
                     'width': intr.width, 'height': intr.height,
                     }

print("相机内参为：\n", camera_parameters)
# 保存内参到本地
# with open('./intrinsics.json', 'w') as fp:
#     json.dump(camera_parameters, fp)
