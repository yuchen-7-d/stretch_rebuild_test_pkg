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

    link_grasp_center_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'link_grasp_center'
    )

    link_ids = {'link_grasp_center':link_grasp_center_id}

    for link_name,link_id in link_ids.items():
        if link_id == -1:
            raise RuntimeError(f'找不到参考点:{link_name}')

    world_link = data.xpos[link_grasp_center_id].copy()

    print(f'参考点ID:{link_grasp_center_id}')
    print(f'世界坐标:{world_link}')
    print(f'世界坐标形状:{world_link.shape}')

    blue_offset = blue - world_link
    blue_distance = np.linalg.norm(blue_offset)
    recovered_point = blue_offset + world_link
    direction_ok = np.allclose(
        recovered_point,
        blue,
        rtol=0.0,
        atol=1e-9
    )

    print(f'夹爪到蓝色目标的位移:{blue_offset}m')
    print(f'位移数组形状:{blue_offset.shape}')
    print(f'夹爪到蓝色目标的直线距离:{blue_distance:.6f}m')
    print(f'位移方向检查:{direction_ok}')

    lift_actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'lift'
    )

    actuator_ids = {'lift':lift_actuator_id}

    for actuator_name,actuator_id in actuator_ids.items():
        if actuator_id == -1:
            raise RuntimeError(f'找不到升降关节:{actuator_name}')

    lift_postion = data.joint('joint_lift').qpos[0]
    lift_target = data.ctrl[lift_actuator_id]
    lift_ctrlrange = model.actuator_ctrlrange[lift_actuator_id]

    print(f'升降关节位置:{lift_postion}')
    print(f'升降关节控制目标:{lift_target}')
    print(f'升降关节控制范围:{lift_ctrlrange}')
    print(f'升降关节控制范围属性:{lift_ctrlrange.shape}')

    lift_delta = 5 / 100
    lift_goal = lift_delta + lift_postion
    lift_lower = lift_ctrlrange[0]
    lift_upper = lift_ctrlrange[1]

    goal_in_range = lift_lower <= lift_goal <= lift_upper

    if not goal_in_range:
        raise RuntimeError(f'升降目标超出范围:{lift_goal}')

    print(f'计划增加量:{lift_delta:.6f}m')
    print(f'新升降目标:{lift_goal:.6f}m')
    print(f'是否在控制范围内:{goal_in_range}')

    motion_start_time = data.time
    motion_duration = 2.0

    lift_before = lift_postion
    grasp_before = world_link.copy()

    data.ctrl[lift_actuator_id] = lift_goal

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

        while (
            windows.is_running()
            and data.time - motion_start_time < motion_duration
        ):
            mujoco.mj_step(model, data)
            windows.sync()
            time.sleep(model.opt.timestep)

    mujoco.mj_forward(model, data)

    lift_after = data.joint('joint_lift').qpos[0]
    grasp_after = data.xpos[link_grasp_center_id].copy()

    motion_elapsed = data.time - motion_start_time
    lift_actual_delta = lift_after - lift_before
    lift_error = lift_after - lift_goal
    grasp_height_delta = grasp_after[2] - grasp_before[2]

    final_lift_speed = data.joint('joint_lift').qvel[0]
    after_lift_goal = data.ctrl[lift_actuator_id]

    lift_gain_params = model.actuator_gainprm[lift_actuator_id].copy()
    lift_bias_params = model.actuator_biasprm[lift_actuator_id].copy()

    print(f'最终升降速度:{final_lift_speed:.6e} m/s')
    print(f'最终升降控制值:{after_lift_goal:.6f} m')
    print(f'升降执行器增益参数:{lift_gain_params}')
    print(f'升降执行器偏置参数:{lift_bias_params}')

    print(f'本次仿真耗时:{motion_elapsed:.6f}s')
    print(f'最终升降位置:{lift_after:.6f}m')
    print(f'实际关节增加量:{lift_actual_delta:.6f}m')
    print(f'升降目标误差:{lift_error:.6f}m')
    print(f'夹爪高度增加量:{grasp_height_delta:.6f}m')

    print(f'关闭窗口时间:{data.time}')
    print(f'蓝色坐标:{blue}')
    print(f'红色坐标:{red}')
    print(f'蓝色坐标形状:{blue.shape}')
    print(f'红色坐标形状:{red.shape}')


if __name__ == '__main__':
    main()
