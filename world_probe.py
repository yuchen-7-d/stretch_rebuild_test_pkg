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

    scene_path = Path('/home/yu/stretch_rebuild/saved_picture/scene_state.npz')
    with np.load(scene_path) as scene:
        if scene['qpos'].shape != data.qpos[:].shape:
            raise RuntimeError('qpos形状不匹配')
        if scene['qvel'].shape != data.qvel[:].shape:
            raise RuntimeError('qvel形状不匹配')
        if scene['ctrl'].shape != data.ctrl[:].shape:
            raise RuntimeError('ctrl形状不匹配')

        data.qpos[:] = scene['qpos']
        data.qvel[:] = scene['qvel']
        data.ctrl[:] = scene['ctrl']
        saved_time = scene['time']
        data.time = float(saved_time)

    mujoco.mj_forward(model, data)

    print(f'打开窗口时间:{data.time}')
    with mujoco.viewer.launch_passive(model, data) as windows:

        with windows.lock():
            mujoco.mjv_initGeom(
                windows.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=blue,
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 1]
            )
            mujoco.mjv_initGeom(
                windows.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=red,
                mat=np.eye(3).flatten(),
                rgba=[1, 1, 0, 1]
            )

            windows.user_scn.ngeom = 2

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