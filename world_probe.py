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

    wrist_pitch_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'wrist_pitch'
    )

    arm_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'arm'
    )

    actuator_ids = {
        'lift':lift_actuator_id,
        'wrist_pitch':wrist_pitch_id,
        'arm':arm_id
    }

    for actuator_name,actuator_id in actuator_ids.items():
        if actuator_id == -1:
            raise RuntimeError(f'找不到:{actuator_name}')

    lift_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        'joint_lift'
    )

    if lift_joint_id == -1:
        raise RuntimeError('找不到升降关节')

    lift_dof_id = model.jnt_dofadr[lift_joint_id]

    print(f'升降关节ID:{lift_joint_id}')
    print(f'升降自由度索引:{lift_dof_id}')

    arm_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        'joint_arm_l0'
    )

    if arm_joint_id == -1:
        raise RuntimeError('找不到伸缩关节')

    lift_postion = data.joint('joint_lift').qpos[0]
    lift_target = data.ctrl[lift_actuator_id]
    lift_ctrlrange = model.actuator_ctrlrange[lift_actuator_id]

    print(f'升降关节位置:{lift_postion}')
    print(f'升降关节控制目标:{lift_target}')
    print(f'升降关节控制范围:{lift_ctrlrange}')
    print(f'升降关节控制范围属性:{lift_ctrlrange.shape}')

    wrist_pitch_position = data.joint('joint_wrist_pitch').qpos[0]
    wrist_pitch_target = data.ctrl[wrist_pitch_id]
    wrist_pitch_ctrlrange = model.actuator_ctrlrange[wrist_pitch_id]

    print(f'手腕关节位置:{wrist_pitch_position}rad')
    print(f'手腕控制目标:{wrist_pitch_target}rad')
    print(f'手腕控制范围:{wrist_pitch_ctrlrange}')
    print(f'手腕控制范围属性:{wrist_pitch_ctrlrange.shape}')

    arm_target = data.ctrl[arm_id]
    arm_ctrlrange = model.actuator_ctrlrange[arm_id]
    actuator_arm_length = data.actuator_length[arm_id]

    print(f'手臂控制目标:{arm_target}')
    print(f'手臂控制范围:{arm_ctrlrange}')
    print(f'手臂实际伸长量:{actuator_arm_length}')

    arm_before = actuator_arm_length
    arm_delta = 0.02
    arm_goal = actuator_arm_length + arm_delta

    arm_in_range = arm_ctrlrange[0] <= arm_goal <= arm_ctrlrange[1]
    if not arm_in_range:
        raise RuntimeError(f'手臂目标超出范围:{arm_goal}')

    arm_duration = 2.0
    arm_tolerance = 0.001

    lift_delta = 0.05
    lift_commpensation = 0.011
    motion_duration = 3.0

    pregrasp_margin = 0.05

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

        arm_start_time = data.time
        data.ctrl[arm_id] = arm_goal

        while(
            windows.is_running()
            and data.time - arm_start_time < arm_duration
        ):
            mujoco.mj_step(model, data)
            windows.sync()
            time.sleep(model.opt.timestep)

        mujoco.mj_forward(model, data)

        arm_after = data.actuator_length[arm_id]
        arm_actual_delta = arm_after - actuator_arm_length
        arm_error = arm_after - arm_goal
        arm_elapsed = data.time - arm_start_time

        print(f'手臂目标:{arm_goal}')
        print(f'最终实际伸长量:{arm_after}')
        print(f'实际增加量:{arm_actual_delta}')
        print(f'目标误差:{arm_error}')
        print(f'阶段耗时:{arm_elapsed}')

        if not windows.is_running():
            print('伸臂阶段窗口关闭，不执行升降')
            return

        if abs(arm_error) > arm_tolerance:
            print('伸臂尚未到位，不执行伸降')
            return

        lift_before = data.joint('joint_lift').qpos[0]
        grasp_before = data.xpos[link_grasp_center_id].copy()

        blue_z = blue[2]
        current_grasp_z = grasp_before[2]

        pregrasp_z = blue_z + pregrasp_margin
        height_gap = pregrasp_z - current_grasp_z

        candidate_lift_goal = lift_before + height_gap
        candidate_lift_command = candidate_lift_goal + lift_commpensation

        candidate_goal_in_range = lift_ctrlrange[0] <= candidate_lift_goal <= lift_ctrlrange[1]
        candidate_command_in_range = lift_ctrlrange[0] <= candidate_lift_command <= lift_ctrlrange[1]

        print(f'蓝色视觉点高度:{blue_z:.6f} m')
        print(f'准备高度:{pregrasp_z:.6f} m')
        print(f'当前夹爪高度:{current_grasp_z:.6f} m')
        print(f'需要增加的高度:{height_gap:.6f} m')
        print(f'候选升降目标:{candidate_lift_goal:.6f} m')
        print(f'候选控制命令:{candidate_lift_command:.6f} m')
        print(f'候选目标是否在范围内:{candidate_goal_in_range}')
        print(f'候选命令是否在范围内:{candidate_command_in_range}')

        lift_goal = candidate_lift_goal
        goal_in_range = lift_ctrlrange[0] <= lift_goal <= lift_ctrlrange[1]
        if not goal_in_range:
            raise RuntimeError(f'伸降目标超出范围:{lift_goal}')

        lift_command = lift_goal + lift_commpensation

        command_in_range = lift_ctrlrange[0] <= lift_command <= lift_ctrlrange[1]
        if not command_in_range:
            raise RuntimeError(f'升降控制命令超出范围:{lift_command}')

        print(f'升降起点:{lift_before}')
        print(f'新伸降目标:{lift_goal}')
        print(f'升降补偿量:{lift_commpensation}')
        print(f'下发升降命令:{lift_command}')

        motion_start_time = data.time
        data.ctrl[lift_actuator_id] = lift_command

        while (
            windows.is_running()
            and data.time - motion_start_time < motion_duration
        ):
            mujoco.mj_step(model, data)
            windows.sync()
            time.sleep(model.opt.timestep)

        mujoco.mj_forward(model, data)

        xaxis_arm_joint = data.xaxis[arm_joint_id].copy()
        arm_joint_distance = np.linalg.norm(xaxis_arm_joint)
        arm_axis_xy = xaxis_arm_joint[:2]

        print(f'伸缩臂世界方向:{xaxis_arm_joint}')
        print(f'方向数组形状:{xaxis_arm_joint.shape}')
        print(f'方向向量长度:{arm_joint_distance}')

        lift_after = data.joint('joint_lift').qpos[0]
        grasp_after = data.xpos[link_grasp_center_id].copy()

        remaining_offset = blue - grasp_after
        horizontal_offset = remaining_offset[:2]
        horizontal_distance = np.linalg.norm(horizontal_offset)

        print(f'最终夹爪世界坐标:{grasp_after}')
        print(f'到蓝色目标的三维位移:{remaining_offset}')
        print(f'三维位移形状;{remaining_offset.shape}')
        print(f'水平位移:{horizontal_offset}m')
        print(f'水平位移形状:{horizontal_offset.shape}')
        print(f'水平距离:{horizontal_distance:.6f}m')

        projection = np.dot(horizontal_offset, arm_axis_xy)
        axis_xy_squared = np.dot(arm_axis_xy, arm_axis_xy)

        if axis_xy_squared < 1e-12:
            raise RuntimeError('伸臂方向的水平分量过小，无法计算')

        planned_arm_delta = projection / axis_xy_squared
        projected_horizontal_move = planned_arm_delta * arm_axis_xy
        residual_horizontal_offset = horizontal_offset - projected_horizontal_move
        residual_horizontal_distance = np.linalg.norm(residual_horizontal_offset)

        print(f'水平伸臂方向:{arm_axis_xy}')
        print(f'水平伸臂方向形状:{arm_axis_xy.shape}')
        print(f'计划伸臂增加量:{planned_arm_delta:.6f}m')
        print(f'预计水平位移:{projected_horizontal_move}m')
        print(f'预计剩余水平偏差:{residual_horizontal_offset}m')
        print(f'预计剩余水平距离:{residual_horizontal_distance:.6f}m')

        approach_arm_start = data.actuator_length[arm_id]
        approach_arm_goal = approach_arm_start + planned_arm_delta
        approach_arm_in_range = (
            arm_ctrlrange[0] <= approach_arm_goal <= arm_ctrlrange[1]
        )
        if not approach_arm_in_range:
            raise RuntimeError(f'接近动作超出范围:{approach_arm_goal}')

        print(f'接近前手臂实际伸长量:{approach_arm_start:.6f}m')
        print(f'计划伸臂增加量:{planned_arm_delta:.6f}m')
        print(f'接近动作目标:{approach_arm_goal:.6f}m')
        print(f'接近目标是否在范围内:{approach_arm_in_range}')

        final_lift_height = grasp_after[2]
        lift_height_error = final_lift_height - pregrasp_z

        print(f'最终爪高度:{final_lift_height}')
        print(f'高度误差:{lift_height_error}')

        motion_elapsed = data.time - motion_start_time
        lift_actual_delta = lift_after - lift_before
        lift_error = lift_after - lift_goal
        grasp_height_delta = grasp_after[2] - grasp_before[2]

        final_lift_speed = data.joint('joint_lift').qvel[0]
        after_lift_goal = data.ctrl[lift_actuator_id]

        lift_gain_params = model.actuator_gainprm[lift_actuator_id].copy()
        lift_bias_params = model.actuator_biasprm[lift_actuator_id].copy()

        lift_force = data.actuator_force[lift_actuator_id].copy()
        lift_forcerange = model.actuator_forcerange[lift_actuator_id].copy()
        lift_forcelimit = model.actuator_forcelimited[lift_actuator_id].copy()

        lift_actuator_force = data.qfrc_actuator[lift_dof_id]
        lift_bias_force = data.qfrc_bias[lift_dof_id]
        lift_passive_force = data.qfrc_passive[lift_dof_id]
        lift_constraint_force = data.qfrc_constraint[lift_dof_id]

        print(f'升降自由度驱动力:{lift_actuator_force:.6f} N')
        print(f'升降动力学偏置项:{lift_bias_force:.6f} N')
        print(f'升降被动力:{lift_passive_force:.6f} N')
        print(f'升降约束力:{lift_constraint_force:.6f} N')

        print(f'当前执行器输出力:{lift_force}')
        print(f'执行器力范围:{lift_forcerange}')
        print(f'是否启用力限制:{lift_forcelimit}')

        print(f'最终升降速度:{final_lift_speed:.6e} m/s')
        print(f'最终升降控制值:{after_lift_goal:.6f} m')
        print(f'升降执行器增益参数:{lift_gain_params}')
        print(f'升降执行器偏置参数:{lift_bias_params}')

        print(f'本次仿真耗时:{motion_elapsed:.6f}s')
        print(f'最终升降位置:{lift_after:.6f}m')
        print(f'实际关节增加量:{lift_actual_delta:.6f}m')
        print(f'升降目标误差:{lift_error:.6f}m')
        print(f'夹爪高度增加量:{grasp_height_delta:.6f}m')

        if not windows.is_running():
            print('窗口已经关闭，不执行接近动作')
            return

        lift_ready = (
            abs(lift_height_error) < 0.005
            and abs(final_lift_speed) < 0.001
        )

        if not lift_ready:
            print('升降高度或速度未达标，不执行接近动作')
            return

        approach_grasp_before = grasp_after.copy()
        approach_start_time = data.time
        approach_duration = 2.0

        data.ctrl[arm_id] = approach_arm_goal

        while(
            windows.is_running()
            and data.time - approach_start_time < approach_duration
        ):
            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            approach_exit_reason = '阶段计时结束'
        else:
            approach_exit_reason = '阶段提前关闭'

        mujoco.mj_forward(model,data)

        approach_arm_after = data.actuator_length[arm_id]
        approach_grasp_after = data.xpos[link_grasp_center_id].copy()

        approach_elapsed = data.time - approach_start_time
        approach_arm_actual_delta = approach_arm_after - approach_arm_start
        approach_arm_error = approach_arm_after - approach_arm_goal

        approach_actual_xy_move = approach_grasp_after[:2] - approach_grasp_before[:2]
        approach_remaining_xy = blue[:2] - approach_grasp_after[:2]
        approach_remaining_distance = np.linalg.norm(approach_remaining_xy)

        approach_height_error = approach_grasp_after[2] - pregrasp_z

        print(f'接近阶段结束原因:{approach_exit_reason}')
        print(f'接近阶段耗时:{approach_elapsed:.6f}s')
        print(f'接近阶段实际伸臂量:{approach_arm_actual_delta:.6f}m')
        print(f'接近阶段手臂目标误差:{approach_arm_error}m')

        print(f'预计水平位移:{projected_horizontal_move}m')
        print(f'实际水平位移:{approach_actual_xy_move}m')
        print(f'实际剩余水平偏差:{approach_remaining_xy}m')
        print(f'预计剩余水平距离:{residual_horizontal_distance:.6f}m')
        print(f'实际剩余水平距离:{approach_remaining_distance:.6f}m')
        print(f'接近后的高度误差:{approach_height_error:.6f}m')

    print(f'关闭窗口时间:{data.time}')
    print(f'蓝色坐标:{blue}')
    print(f'红色坐标:{red}')
    print(f'蓝色坐标形状:{blue.shape}')
    print(f'红色坐标形状:{red.shape}')


if __name__ == '__main__':
    main()
