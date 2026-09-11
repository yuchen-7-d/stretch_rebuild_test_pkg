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

    head_pan_actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "head_pan"
    )

    actuator_ids = {"head_pan": head_pan_actuator_id}

    for actuator_name, actuator_id in actuator_ids.items():
        if actuator_id == -1:
            raise RuntimeError(f'找不到Mujoco actuator: {actuator_name}')
        print(
            f'actuator name = {actuator_name}', 
            f'actuator id = {actuator_id}'
        )

    head_pan_ctrl_range = model.actuator_ctrlrange[
        head_pan_actuator_id
    ]

    target_angle = 0.3

    lower_limit, upper_limit = head_pan_ctrl_range

    if not (lower_limit <= target_angle <=upper_limit):
        raise ValueError(f'目标角度超出控制范围:{target_angle}')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[head_pan_actuator_id] = target_angle
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print(f'推进后时间:{data.time:.6f}s')

    joint_state = data.joint('joint_head_pan')
    actual_angle = joint_state.qpos[0]
    angle_error = actual_angle - target_angle

    print(f'目标角度:{target_angle:.6f} rad')
    print(f'实际角度:{actual_angle:.6f} rad')
    print(f'角度误差:{angle_error:.6f} rad')


if __name__ == '__main__':
    main()
