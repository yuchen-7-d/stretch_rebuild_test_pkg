from pathlib import Path

import mujoco
import cv2 as cv


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

    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    image_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_rgb.png')
    saved = cv.imwrite(str(image_path), bgr_image)

    if not saved:
        raise RuntimeError(f'图片保存失败:{image_path}')

    print(f'图像状态:{rgb_image.shape}')
    print(f'图像类型:{rgb_image.dtype}')
    print(f'保存状态:{saved}')
    print(f'保存至:{image_path}')


if __name__ == '__main__':
    main()
