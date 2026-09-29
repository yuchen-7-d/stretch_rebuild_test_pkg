from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time


def main():
    load_point_path = Path('/home/yu/stretch_rebuild/saved_picture/target_points.npz')
    with np.load(load_point_path) as points:
        blue = points['blue']
        red = points['red']

    camera_probe_path = Path(
        "/home/yu/stretch_projects/stretch_mujoco/"
        "stretch_mujoco/models/scene.xml"
    )

    if not camera_probe_path.is_file():
        raise FileNotFoundError(f'找不到场景文件:{camera_probe_path}')

    model = mujoco.MjModel.from_xml_path(str(camera_probe_path))
    data = mujoco.MjData(model)

    data.joint('joint_head_pan').qpos[0] = -1.57
    data.joint('joint_head_tilt').qpos[0] = -0.9

    mujoco.mj_forward(model, data)

    print(f'打开窗口时间:{data.time}')
    with mujoco.viewer.launch_passive(model, data) as windows:
        while windows.is_running():
            windows.sync()
            time.sleep(0.01)

    print(f'关闭窗口时间:{data.time}')
    print(f'蓝色坐标:{blue}')
    print(f'红色坐标:{red}')
    print(f'蓝色坐标形状:{blue.shape}')
    print(f'红色坐标形状:{red.shape}')


if __name__ == '__main__':
    main()