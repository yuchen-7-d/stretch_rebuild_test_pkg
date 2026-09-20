import cv2 as cv
import numpy as np

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