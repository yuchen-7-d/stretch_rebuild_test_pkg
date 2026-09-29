from pathlib import Path

import mujoco
import cv2 as cv
import numpy as np
import math


def main():
    camera_probe_path = Path(
        "/home/yu/stretch_projects/stretch_mujoco/"
        "stretch_mujoco/models/scene.xml"
    )

    if not camera_probe_path.is_file():
        raise FileNotFoundError(f'找不到场景文件:{camera_probe_path}')

    model = mujoco.MjModel.from_xml_path(str(camera_probe_path))
    data = mujoco.MjData(model)

    d435i_camera_rgb_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "d435i_camera_rgb"
    )

    camera_ids = {"d435i_camera_rgb":d435i_camera_rgb_id}

    for camera_name, camera_id in camera_ids.items():
        if camera_id == -1:
            raise RuntimeError(f'找不到Mj_name:{camera_name}')

    print(f'相机数量:{model.ncam}')
    print(f'目标相机:{camera_name}')
    print(f'相机ID:{d435i_camera_rgb_id}')

    data.joint('joint_head_pan').qpos[0] = -1.57
    data.joint('joint_head_tilt').qpos[0] = -0.9

    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera=d435i_camera_rgb_id)
        rgb_image = renderer.render()
        renderer.enable_depth_rendering()
        depth_image = renderer.render()

    rgb_image_height = rgb_image.shape[0]
    rgb_image_width = rgb_image.shape[1]
    fovy_deg = model.cam_fovy[d435i_camera_rgb_id]
    fovy_rad = np.deg2rad(fovy_deg)

    fy = rgb_image_height / (2 * math.tan (fovy_rad/2))
    fx = fy
    cx = (rgb_image_width - 1) / 2
    cy = (rgb_image_height - 1) / 2

    depth_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_depth.npy')
    np.save(depth_path, depth_image)

    loaded_depth = np.load(depth_path)

    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    image_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_rgb.png')
    image_saved = cv.imwrite(str(image_path), bgr_image)

    if not image_saved:
        raise RuntimeError(f'图片保存失败:{image_path}')

    print(f'图像状态:{rgb_image.shape}')
    print(f'图像类型:{rgb_image.dtype}')
    print(f'图像宽度:{rgb_image_width}')
    print(f'图像高度:{rgb_image_height}')
    print(f'垂直视场角:{fovy_deg} deg')
    print(f'fx:{fx}')
    print(f'fy:{fy}')
    print(f'cx:{cx}')
    print(f'cy:{cy}')
    print(f'保存状态:{image_saved}')
    print(f'保存至:{image_path}')
    print(f'深度图形状:{depth_image.shape}')
    print(f'深度图类型:{depth_image.dtype}')
    print(f'最小深度:{depth_image.min():.6f}m')
    print(f'最大深度:{depth_image.max():.6f}m')
    print(f'深度图保存路径:{depth_path}')
    print(f'加载深度图形状:{loaded_depth.shape}')
    print(f'加载深度图类型:{loaded_depth.dtype}')
    print(f'保存前后数组是否一致:{np.array_equal(loaded_depth, depth_image)}')


if __name__ == '__main__':
    main()
