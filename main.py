from pathlib import Path

import mujoco
import mujoco.viewer
import time


def main():
    scene_path = Path(
        "/home/yu/stretch_projects/stretch_mujoco/"
        "stretch_mujoco/models/scene.xml"
    )

    if not scene_path.is_file():
        raise FileNotFoundError(f'找不到场景文件:{scene_path}')

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    print(f'场景文件:{scene_path}')
    print(f'关节数量:{model.njnt}')
    print(f'执行器数量:{model.nu}')

    data = mujoco.MjData(model)
    start_time = data.time
    print(f'推进前时间:{start_time:.6f}s')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print(f'推进后时间:{data.time:.6f}s')


if __name__ == '__main__':
    main()
