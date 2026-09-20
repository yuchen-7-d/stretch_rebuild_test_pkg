import cv2 as cv
import numpy as np

image_read = cv.imread('/home/yu/stretch_rebuild/saved_picture/camera_rgb.png')

if not image_read:
    raise RuntimeError('未读取图片')

hsv_image = cv.cvtColor(image_read, cv.COLOR_BGR2HSV)