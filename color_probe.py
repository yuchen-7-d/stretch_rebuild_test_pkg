import cv2 as cv
import numpy as np


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