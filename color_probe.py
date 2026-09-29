import cv2 as cv
import numpy as np
from pathlib import Path


def get_object_info(mask):
    contours, hierarchy= cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    target_contour = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(target_contour)

    center_x = x + w * 0.5
    center_y = y + h * 0.5

    return {
        'bbox': (x, y, w, h),
        'center': (center_x, center_y),
    }

image_read = cv.imread('/home/yu/stretch_rebuild/saved_picture/camera_rgb.png')

if image_read is None:
    raise RuntimeError(f'未读取图片:{image_read}')

camera_depth_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_depth.npy')
loaded_camera_depth = np.load(camera_depth_path)

if loaded_camera_depth.shape[:2] != image_read.shape[:2]:
    raise RuntimeError(f'{image_read.shape}尺寸不一致')

intrinsics_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_intrinsics.npz')
with np.load(intrinsics_path) as intrinsics:
    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    intrinsics_width = int(intrinsics['width'])
    intrinsics_height = int(intrinsics['height'])

if(intrinsics_height, intrinsics_width) != image_read.shape[:2]:
    raise RuntimeError('内参对应的图像尺寸与当前图片不一致')

print(f'内参图像宽度:{intrinsics_width}')
print(f'内参图像高度:{intrinsics_height}')
print(f'读取fx:{fx:.6f}')
print(f'读取fy:{fy:.6f}')
print(f'读取cx:{cx:.6f}')
print(f'读取cy:{cy:.6f}')

pose_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_pose.npz')
with np.load(pose_path) as pose:
    position = pose['position']
    rotation = pose['rotation']

if position.shape != (3,):
    raise RuntimeError('相机位置状态错误')
if rotation.shape != (3, 3):
    raise RuntimeError('相机角度状态错误')

hsv_image = cv.cvtColor(image_read, cv.COLOR_BGR2HSV)

lower_blue = np.array([100, 80, 40], dtype=np.uint8)
upper_blue = np.array([130, 255, 255], dtype=np.uint8)
lower_red_first = np.array([0, 80, 40], dtype=np.uint8)
upper_red_first = np.array([10, 255, 255], dtype=np.uint8)
lower_red_scened = np.array([170, 80, 40], dtype=np.uint8)
upper_red_secend = np.array([179, 255, 255], dtype=np.uint8)

red_mask_frist = cv.inRange(hsv_image, lower_red_first, upper_red_first)
red_mask_secend = cv.inRange(hsv_image, lower_red_scened, upper_red_secend)

blue_mask = cv.inRange(hsv_image, lower_blue, upper_blue)
red_mask = cv.bitwise_or(red_mask_frist, red_mask_secend)

blue_mask_path = '/home/yu/stretch_rebuild/saved_picture/blue_mask.png'
saved_blue = cv.imwrite(str(blue_mask_path), blue_mask)
red_mask_path = '/home/yu/stretch_rebuild/saved_picture/red_mask.png'
saved_red = cv.imwrite(str(red_mask_path), red_mask)

if not saved_blue:
    raise RuntimeError(f'蓝色掩膜保存失败:{saved_blue}')
if not saved_red:
    raise RuntimeError(f'红色掩膜保存失败:{saved_red}')

blue_pixel_count = cv.countNonZero(blue_mask)
red_pixel_count = cv.countNonZero(red_mask)

print(f'蓝色掩膜形状:{blue_mask.shape}')
print(f'蓝色掩膜类型:{blue_mask.dtype}')
print(f'蓝色像素数量:{blue_pixel_count}')
print(f'蓝色是否保存成功:{saved_blue}')
print(f'红色掩膜形状:{red_mask.shape}')
print(f'红色掩膜类型:{red_mask.dtype}')
print(f'红色像素数量:{red_pixel_count}')
print(f'红色是否保存成功:{saved_red}')

blue_info = get_object_info(blue_mask)
red_info = get_object_info(red_mask)

annotated_image = image_read.copy()

if blue_info is None:
    print('未找到蓝色目标')
else:
    print(f"蓝色边框:{blue_info['bbox']}")
    print(f"蓝色中心:{blue_info['center']}")
    x, y, w, h = blue_info['bbox']
    center_x, center_y = blue_info['center']

    pixel_x = round(center_x)
    pixel_y = round(center_y)

    blue_depth = loaded_camera_depth[pixel_y, pixel_x]
    print(f'蓝色中心像素坐标:({pixel_x}, {pixel_y})')
    print(f'蓝色中心深度:{blue_depth:.6f}m')

    blue_camera_Z = blue_depth
    blue_camera_X = ((pixel_x - cx) * blue_camera_Z) / fx
    blue_camera_Y = ((pixel_y - cy) * blue_camera_Z) / fy
    print(f'蓝色相机Z坐标:{blue_camera_Z:.6f}m')
    print(f'蓝色相机X坐标:{blue_camera_X:.6f}m')
    print(f'蓝色相机Y坐标:{blue_camera_Y:.6f}m')

    blue_mujoco_point = np.array(
        [blue_camera_X, -blue_camera_Y, -blue_camera_Z],
        dtype=np.float64
    )
    blue_world_point = rotation @ blue_mujoco_point + position
    print(f'蓝色相机mujoco坐标:{blue_mujoco_point}')
    print(f'蓝色相机世界坐标:{blue_world_point}')
    print(f'蓝色相机世界坐标形状:{blue_world_point.shape}')

    cv.rectangle(
        annotated_image,
        (x, y),
        (x + w - 1, y + h - 1),
        (0, 255, 0),
        thickness=2
    )
    cv.circle(
        annotated_image,
        (round(center_x), round(center_y)),
        radius=4,
        color=(0, 255, 255),
        thickness=-1
    )

if red_info is None:
    print('未找到红色目标')
else:
    print(f"红色边框:{red_info['bbox']}")
    print(f"红色中心:{red_info['center']}")
    x, y, w, h = red_info['bbox']
    center_x, center_y = red_info['center']

    pixel_x = round(center_x)
    pixel_y = round(center_y)

    red_depth = loaded_camera_depth[pixel_y, pixel_x]
    print(f'红色中心像素坐标:({pixel_x}, {pixel_y})')
    print(f'红色中心深度:{red_depth:.6f}m')

    red_camera_Z = red_depth
    red_camera_X = ((pixel_x - cx) * red_camera_Z) / fx
    red_camera_Y = ((pixel_y - cy) * red_camera_Z) / fy
    print(f'红色相机Z坐标:{red_camera_Z:.6f}m')
    print(f'红色相机X坐标:{red_camera_X:.6f}m')
    print(f'红色相机Y坐标:{red_camera_Y:.6f}m')

    red_mujoco_point = np.array(
        [red_camera_X, -red_camera_Y, -red_camera_Z],
        dtype=np.float64
    )
    red_world_point = rotation @ red_mujoco_point + position
    print(f'红色相机mujoco坐标:{red_mujoco_point}')
    print(f'红色相机世界坐标:{red_world_point}')
    print(f'红色相机世界坐标形状:{red_world_point.shape}')

    cv.rectangle(
        annotated_image,
        (x, y),
        (x + w - 1, y + h - 1),
        (0, 255, 0),
        thickness=2
    )
    cv.circle(
        annotated_image,
        (round(center_x), round(center_y)),
        radius=4,
        color=(0, 255, 255),
        thickness=-1
    )

detection_path = '/home/yu/stretch_rebuild/saved_picture/detection.png'
saved_detection = cv.imwrite(str(detection_path), annotated_image)

if not saved_detection:
    raise RuntimeError(f'标注图片保存失败:{detection_path}')

print(f'标注图片保存是否成功:{saved_detection}')
print(f'保存至:{detection_path}')