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
    strat_time = data.time

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


if __name__ == '__main__':
    main()