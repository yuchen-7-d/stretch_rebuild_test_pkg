from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time


def main():

    target_color = input('请选择抓取颜色:(red/blue):').strip().lower()

    if target_color not in('blue','red'):
        raise RuntimeError(f'不支持的颜色:{target_color}')

    load_point_path = Path(
        '/home/yu/stretch_rebuild/saved_picture/target_points.npz'
    )

    with np.load(load_point_path) as points:
        blue = points['blue']
        red = points['red']

        target_point = points[target_color]

    print(f'选择的颜色:{target_color}')
    print(f'目标世界坐标:{target_point}')
    print(f'目标坐标形状:{target_point.shape}')

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

    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'base_link'
    )

    left_rubber_tip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'rubber_tip_left'
    )

    right_rubber_tip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'rubber_tip_right'
    )

    if target_color == 'blue':
        target_body_name = 'object1'
    else:
        target_body_name = 'object2'

    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        target_body_name
    )

    link_ids = {
        'link_grasp_center':link_grasp_center_id,
        'base_link':base_body_id,
        'rubber_tip_left':left_rubber_tip_id,
        'rubber_tip_right':right_rubber_tip_id,
        target_body_name:target_body_id
    }

    for link_name,link_id in link_ids.items():
        if link_id == -1:
            raise RuntimeError(f'找不到:{link_name}')

    world_link = data.xpos[link_grasp_center_id].copy()

    print(f'参考点ID:{link_grasp_center_id}')
    print(f'世界坐标:{world_link}')
    print(f'世界坐标形状:{world_link.shape}')

    target_offset = target_point - world_link
    target_distance = np.linalg.norm(target_offset)
    recovered_point = target_offset + world_link
    direction_ok = np.allclose(
        recovered_point,
        target_point,
        rtol=0.0,
        atol=1e-9
    )

    print(f'夹爪到目标的位移:{target_offset}m')
    print(f'位移数组形状:{target_offset.shape}')
    print(f'夹爪到目标的直线距离:{target_distance:.6f}m')
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

    left_wheel_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'left_wheel_vel'
    )

    right_wheel_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'right_wheel_vel'
    )

    gripper_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'gripper'
    )

    actuator_ids = {
        'lift':lift_actuator_id,
        'wrist_pitch':wrist_pitch_id,
        'arm':arm_id,
        'left_wheel_vel':left_wheel_id,
        'right_wheel_vel':right_wheel_id,
        'gripper':gripper_id
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

    arm_delta = 0.02
    arm_goal = actuator_arm_length + arm_delta

    arm_in_range = arm_ctrlrange[0] <= arm_goal <= arm_ctrlrange[1]
    if not arm_in_range:
        raise RuntimeError(f'手臂目标超出范围:{arm_goal}')

    arm_duration = 2.0
    arm_tolerance = 0.001

    lift_commpensation = 0.011
    motion_duration = 3.0

    pregrasp_margin = 0.05

    left_wheel_ctrl = data.ctrl[left_wheel_id]
    left_wheel_ctrlrange = model.actuator_ctrlrange[left_wheel_id]
    left_wheel_gear = model.actuator_gear[left_wheel_id]

    right_wheel_ctrl = data.ctrl[right_wheel_id]
    right_wheel_ctrlrange = model.actuator_ctrlrange[right_wheel_id]
    right_wheel_gear = model.actuator_gear[right_wheel_id]

    print(f'左轮ID:{left_wheel_id}')
    print(f'左轮控制值:{left_wheel_ctrl}')
    print(f'左轮控制值属性:{left_wheel_ctrl.shape}')
    print(f'左轮控制范围:{left_wheel_ctrlrange}')
    print(f'左轮传动参数:{left_wheel_gear}')
    print(f'右轮ID:{right_wheel_id}')
    print(f'右轮控制值:{right_wheel_ctrl}')
    print(f'右轮控制值属性:{right_wheel_ctrl.shape}')
    print(f'右轮控制范围:{right_wheel_ctrlrange}')
    print(f'右轮传动参数:{right_wheel_gear}')

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

        target_z = target_point[2]
        current_grasp_z = grasp_before[2]

        pregrasp_z = target_z + pregrasp_margin
        height_gap = pregrasp_z - current_grasp_z

        candidate_lift_goal = lift_before + height_gap
        candidate_lift_command = candidate_lift_goal + lift_commpensation

        candidate_goal_in_range = lift_ctrlrange[0] <= candidate_lift_goal <= lift_ctrlrange[1]
        candidate_command_in_range = lift_ctrlrange[0] <= candidate_lift_command <= lift_ctrlrange[1]

        print(f'目标视觉点高度:{target_z:.6f} m')
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

        remaining_offset = target_point - grasp_after
        horizontal_offset = remaining_offset[:2]
        horizontal_distance = np.linalg.norm(horizontal_offset)

        print(f'最终夹爪世界坐标:{grasp_after}')
        print(f'到目标的三维位移:{remaining_offset}')
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
        approach_remaining_xy = target_point[:2] - approach_grasp_after[:2]
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

        base_rotation = data.xmat[base_body_id].reshape(3, 3).copy()
        base_forward_world = base_rotation[:,0].copy()
        base_forward_length = np.linalg.norm(base_forward_world)

        print(f'底座ID:{base_body_id}')
        print(f'底座朝向矩阵形状:{base_rotation.shape}')
        print(f'底座朝向世界方向:{base_forward_world}')
        print(f'方向数组形状:{base_forward_world.shape}')
        print(f'方向向量长度:{base_forward_length}')

        base_forward_xy = base_forward_world[:2].copy()
        base_forward_xy_norm = np.linalg.norm(base_forward_xy)

        if base_forward_xy_norm < 1e-12:
            raise RuntimeError('底座水平朝前方向长度过小')

        base_unit_xy = base_forward_xy / base_forward_xy_norm

        planned_base_distance = np.dot(base_unit_xy, approach_remaining_xy)

        base_predicted_xy_move = planned_base_distance * base_unit_xy

        base_remaining_xy = approach_remaining_xy - base_predicted_xy_move
        base_remaining_distance = np.linalg.norm(base_remaining_xy)

        print(f'底座水平单位方向:{base_unit_xy}')
        print(f'水平单位方向形状:{base_unit_xy.shape}')
        print(f'水平单位方向长度:{np.linalg.norm(base_unit_xy):.6f}')
        print(f'计划底座移动距离:{planned_base_distance:.6f}m')
        print(f'底座预计水平位移:{base_predicted_xy_move}m')
        print(f'底座移动后预计剩余偏差:{base_remaining_xy}m')
        print(f'底座移动后预计剩余距离:{base_remaining_distance:.6f}m')

        base_speed = 0.05
        wheel_radius = 0.05

        wheel_angular_speed = base_speed / wheel_radius

        left_wheel_command = left_wheel_gear[0] * wheel_angular_speed
        right_wheel_command = right_wheel_gear[0] * wheel_angular_speed

        left_command_in_range = (
            left_wheel_ctrlrange[0] <= left_wheel_command <= left_wheel_ctrlrange[1]
        )
        if not left_command_in_range:
            raise RuntimeError('左轮命令超出范围')
        right_command_in_range = (
            right_wheel_ctrlrange[0] <= right_wheel_command <= right_wheel_ctrlrange[1]
        )
        if not right_command_in_range:
            raise RuntimeError('右轮命令超出范围')

        print(f'底座速度目标:{base_speed:.6f} m/s')
        print(f'车轮角速度目标:{wheel_angular_speed:.6f} rad/s')
        print(f'左轮控制命令:{left_wheel_command:.6f}')
        print(f'右轮控制命令:{right_wheel_command:.6f}')
        print(f'左轮命令是否在范围内:{left_command_in_range}')
        print(f'右轮命令是否在范围内:{right_command_in_range}')

        if not windows.is_running():
            print('窗口已关闭，不执行底座移动')
            return

        approach_ready = (
            abs(approach_arm_error) < 0.001
            and abs(approach_height_error) < 0.005
        )

        if not approach_ready:
            print('接近阶段还没准备好，不移动')
            return

        if planned_base_distance < 0:
            raise RuntimeError('本阶段尚未处理倒车目标')

        base_tolerance = 0.001
        base_timeout = 5.0

        mujoco.mj_forward(model,data)

        base_start_xy = data.xpos[base_body_id][:2].copy()
        base_start_time = data.time

        base_exit_reason = '未开始'

        try:
            while True:
                mujoco.mj_forward(model, data)
                base_current_xy = data.xpos[base_body_id][:2].copy()

                base_actual_xy_move = base_current_xy - base_start_xy
                base_traveled = np.dot(base_actual_xy_move, base_unit_xy)
                base_remaining = planned_base_distance - base_traveled
                base_elapsed = data.time - base_start_time

                if not windows.is_running():
                    base_exit_reason = '窗口提前关闭'
                    break

                if base_remaining <= base_tolerance:
                    base_exit_reason = '达到停车阀值'
                    break

                if base_elapsed >= base_timeout:
                    base_exit_reason = '超时'
                    break

                data.ctrl[left_wheel_id] = left_wheel_command
                data.ctrl[right_wheel_id] = right_wheel_command

                mujoco.mj_step(model, data)
                windows.sync()
                time.sleep(model.opt.timestep)

        finally:
            data.ctrl[left_wheel_id] = 0.0
            data.ctrl[right_wheel_id] = 0.0

        print(f'底盘前进阶段结束原因:{base_exit_reason}')
        print(f'底盘前进耗时:{base_elapsed:.6f}s')
        print(f'下发停车命令前前进距离:{base_traveled:.6f}m')
        print(f'下发停车命令时的剩余距离:{base_remaining:.6f}m')

        base_velocity = np.zeros(6, dtype=np.float64)

        base_linear_tol = 0.001
        base_angular_tol = 0.01
        base_stable_duration = 0.2
        base_settle_timeout = 2.0

        base_settle_start = data.time
        base_stable_since = None
        base_settled = False
        base_settle_reason = '未开始'

        while True:
            mujoco.mj_forward(model,data)

            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                base_body_id,
                base_velocity,
                0
            )

            base_angular_speed = np.linalg.norm(base_velocity[:3])
            base_linear_speed = np.linalg.norm(base_velocity[3:])
            base_settle_elapsed = data.time - base_settle_start

            if not windows.is_running():
                base_settle_reason = '窗口提前关闭'
                break

            base_slow = (
                base_linear_speed < base_linear_tol
                and base_angular_speed < base_angular_tol
            )

            if base_slow:
                if base_stable_since is None:
                    base_stable_since = data.time

                if data.time - base_stable_since >= base_stable_duration:
                    base_settled = True
                    base_settle_reason = '连续低速达到要求'
                    break

            else:
                base_stable_since = None

            if base_settle_elapsed >= base_settle_timeout:
                base_settle_reason = '等待超时'
                break

            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        print(f'底座停稳是否通过:{base_settled}')
        print(f'停稳检查结束原因:{base_settle_reason}')
        print(f'停稳等待耗时:{base_settle_elapsed:.6f}s')
        print(f'底盘最终线速度:{base_linear_speed:.6e}m/s')
        print(f'底盘最终角速度:{base_angular_speed:.6e}rad/s')
        print(f'左轮最终控制值:{data.ctrl[left_wheel_id]}')
        print(f'右轮最终控制值:{data.ctrl[right_wheel_id]}')

        if not base_settled:
            print('底座未确认停稳，本阶段尚未通过')
            return

        grasp_after_stop = data.xpos[link_grasp_center_id].copy()

        finnal_horizontal_offset = target_point[:2] - grasp_after_stop[:2]
        finnal_horizontal_distance = np.linalg.norm(finnal_horizontal_offset)
        finnal_height_error = grasp_after_stop[2] - pregrasp_z

        print(f'停稳后夹爪世界坐标:{grasp_after_stop}')
        print(f'停稳后夹爪水平偏差:{finnal_horizontal_offset}m')
        print(f'停稳后夹爪水平距离:{finnal_horizontal_distance:.6f}m')
        print(f'停稳后夹爪高度误差:{finnal_height_error:.6f}m')

        glignment_ready = (
            base_exit_reason == '达到停车阀值'
            and base_settled
            and finnal_horizontal_distance < 0.005
            and abs(finnal_height_error) < 0.005
        )

        print(f'抓取准备位置是否验收通过{glignment_ready}')

        if not glignment_ready:
            print('未准备好抓取，不动作夹爪')
            return

        gripper_ctrl = data.ctrl[gripper_id]
        gripper_ctrlrange = model.actuator_ctrlrange[gripper_id]
        gripper_position = data.joint('joint_gripper_slide').qpos[0]

        print(f'夹爪执行器ID:{gripper_id}')
        print(f'夹爪当前控制目标:{gripper_ctrl:.6f}m')
        print(f'夹爪控制范围:{gripper_ctrlrange}')
        print(f'夹爪控制范围形状:{gripper_ctrlrange.shape}')
        print(f'夹爪实际滑动位置:{gripper_position:.6f}m')

        if not windows.is_running():
            print('窗口已关闭，不执行夹爪试探动作')
            return

        mujoco.mj_forward(model,data)

        left_tip_before = data.xpos[left_rubber_tip_id].copy()
        right_tip_before = data.xpos[right_rubber_tip_id].copy()
        tip_distance_before = np.linalg.norm(left_tip_before - right_tip_before)

        gripper_before = data.joint('joint_gripper_slide').qpos[0]
        gripper_delta = 0.01
        gripper_goal = gripper_before + gripper_delta

        gripper_in_range = (
            gripper_ctrlrange[0] <= gripper_goal <= gripper_ctrlrange[1] 
        )

        if not gripper_in_range:
            raise RuntimeError('夹爪试探目标超出控制范围')

        gripper_duration = 1.0
        gripper_start_time = data.time

        data.ctrl[gripper_id] = gripper_goal

        while(
            windows.is_running()
            and data.time - gripper_start_time < gripper_duration
        ):
            mujoco.mj_step(model, data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            gripper_exit_reason = '阶段计时结束'
        else:
            gripper_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)
        gripper_after = data.joint('joint_gripper_slide').qpos[0]

        left_tip_after = data.xpos[left_rubber_tip_id].copy()
        right_tip_after = data.xpos[right_rubber_tip_id].copy()
        tip_distance_after = np.linalg.norm(left_tip_after - right_tip_after)

        gripper_error = gripper_after - gripper_goal
        tip_distance_delta = tip_distance_after - tip_distance_before
        gripper_elapsed = data.time - gripper_start_time

        print(f'夹爪动作结束原因:{gripper_exit_reason}')
        print(f'夹爪动作耗时:{gripper_elapsed:.6f}s')
        print(f'夹爪动作目标:{gripper_goal:.6f}m')
        print(f'夹爪最终实际位置:{gripper_after:.6f}m')
        print(f'夹爪目标误差:{gripper_error:.6f}m')
        print(f'动作前指尖参考点间距:{tip_distance_before:.6f}m')
        print(f'动作后指尖参考点间距:{tip_distance_after:.6f}m')
        print(f'间距变化量:{tip_distance_delta:.6f}m')

        tip_midpoint = (left_tip_after + right_tip_after) / 2

        grasp_reference_now = data.xpos[link_grasp_center_id].copy()

        tip_midpoint_offset = tip_midpoint - grasp_reference_now

        tip_height_above_target = tip_midpoint[2] - target_point[2]

        print(f'张开口左指参考点:{left_tip_after}')
        print(f'张开口右指参考点:{right_tip_after}')
        print(f'两指间参考中点:{tip_midpoint}')
        print(f'中点形状:{tip_midpoint.shape}')
        print(f'中点相对夹爪参考点的偏移:{tip_midpoint_offset}')
        print(f'中点高出目标视觉点:{tip_height_above_target}')

        fine_offset_xy = target_point[:2] - tip_midpoint[:2]

        fine_axis_xy = data.xaxis[arm_joint_id][:2].copy()
        fine_axis_squared = np.dot(fine_axis_xy, fine_axis_xy)

        if fine_axis_squared < 1e-12:
            raise RuntimeError('伸臂方向的水平分量过小')

        fine_arm_delta = (
            np.dot(fine_axis_xy, fine_offset_xy) / fine_axis_squared
        )

        fine_predicted_xy_move = fine_axis_xy * fine_arm_delta
        fine_remaining_xy = fine_offset_xy - fine_predicted_xy_move
        fine_remaining_distance = np.linalg.norm(fine_remaining_xy)

        fine_arm_start = data.actuator_length[arm_id]
        fine_arm_goal = fine_arm_start + fine_arm_delta

        fine_arm_in_range = (
            arm_ctrlrange[0] <= fine_arm_goal <= arm_ctrlrange[1]
        )

        if not fine_arm_in_range:
            raise RuntimeError('伸臂微调目标超出控制范围')

        print(f'指尖中点到目标的水平偏差:{fine_offset_xy}m')
        print(f'当前伸臂水平轴向:{fine_axis_xy}')
        print(f'微调前实际伸长量:{fine_arm_start:.6f}m')
        print(f'计划微调增加量:{fine_arm_delta:.6f}m')
        print(f'微调后的手臂目标:{fine_arm_goal:.6f}m')
        print(f'目标是否在控制范围内:{fine_arm_in_range}')
        print(f'预计微调后剩余水平距离:{fine_remaining_distance:.6f}m')

        if not windows.is_running():
            print('窗口已关闭，不执行伸臂微调')
            return

        gripper_ready = (
            gripper_exit_reason == '阶段计时结束'
            and abs(gripper_error) < 0.001
        )

        if not gripper_ready:
            print('夹爪试探动作未达标，不执行伸臂微调')
            return

        fine_midpoint_before = tip_midpoint.copy()

        fine_duration = 2.0
        fine_start_time = data.time

        data.ctrl[arm_id] = fine_arm_goal

        while(
            windows.is_running()
            and data.time - fine_start_time < fine_duration
        ):
            mujoco.mj_step(model, data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            fine_exit_reason = '阶段计时结束'
        else:
            fine_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)

        fine_arm_after = data.actuator_length[arm_id]

        fine_left_tip_after = data.xpos[left_rubber_tip_id].copy()
        fine_right_tip_after = data.xpos[right_rubber_tip_id].copy()
        fine_midpoint_after = (fine_left_tip_after + fine_right_tip_after) / 2

        fine_actual_delta = fine_arm_after - fine_arm_start
        fine_arm_error = fine_arm_after - fine_arm_goal

        fine_actual_remaining_xy = target_point[:2] - fine_midpoint_after[:2]
        fine_actual_remaining_distance = np.linalg.norm(fine_actual_remaining_xy)

        fine_height_change = fine_midpoint_after[2] - fine_midpoint_before[2]
        fine_elapsed = data.time - fine_start_time

        print(f'伸臂微调结束原因:{fine_exit_reason}')
        print(f'伸臂微调耗时:{fine_elapsed}')
        print(f'微调后实际伸长量:{fine_arm_after}')
        print(f'实际伸臂增加量:{fine_actual_delta}')
        print(f'手臂目标误差:{fine_arm_error:.6f}')
        print(f'微调后指尖中点:{fine_midpoint_after}')
        print(f'微调后实际水平偏差:{fine_actual_remaining_xy}')
        print(f'微调后实际水平距离:{fine_actual_remaining_distance}')
        print(f'微调期间中点高度变化:{fine_height_change}')

        grasp_depth = 0.02
        grasp_midpoint_z = target_point[2] - grasp_depth

        descend_midpoint_start_z = fine_midpoint_after[2]
        descend_delta_z = grasp_midpoint_z - descend_midpoint_start_z

        descend_lift_start = data.joint('joint_lift').qpos[0]

        descend_lift_goal = descend_lift_start + descend_delta_z
        down_lift_compensation = 0.0038
        descend_lift_command = descend_lift_goal + down_lift_compensation

        descend_goal_in_range = (
            lift_ctrlrange[0] <= descend_lift_goal <= lift_ctrlrange[1]
        )

        if not descend_goal_in_range:
            raise RuntimeError('候选升降目标不在范围内')

        descend_command_in_range = (
            lift_ctrlrange[0] <= descend_lift_command <= lift_ctrlrange[1]
        )

        if not descend_command_in_range:
            raise RuntimeError('候选升降命令不再范围内')

        print(f'目标中点高度:{grasp_midpoint_z:.6f}m')
        print(f'当前中点高度:{descend_midpoint_start_z:.6f}m')
        print(f'计划高度变化:{descend_delta_z:.6f}m')
        print(f'下降前升降位置:{descend_lift_start:.6f}m')
        print(f'候选升降目标:{descend_lift_goal:.6f}m')
        print(f'候选升降命令:{descend_lift_command:.6f}m')
        print(f'候选升降目标是否在范围内:{descend_goal_in_range}')
        print(f'候选升降命令是否在范围内:{descend_command_in_range}')

        if not windows.is_running():
            print('窗口已关闭，不执行下降任务')
            return

        fine_arm_ready = (
            fine_exit_reason == '阶段计时结束'
            and abs(fine_arm_error) < 0.001
            and fine_actual_remaining_distance < 0.005
            and abs(fine_height_change) < 0.005
        )

        if not fine_arm_ready:
            print('伸臂微调未结束，不执行下降命令')
            return

        down_duration = 3.0
        down_start_time = data.time

        data.ctrl[lift_actuator_id] = descend_lift_command

        while(
            windows.is_running()
            and data.time - down_start_time < down_duration
        ):
            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            down_exit_reason = '阶段计时结束'
        else:
            down_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)

        print(f'下降后升降控制命令:{data.ctrl[lift_actuator_id]:.6f} m')
        print(f'下降后执行器输出力:{data.actuator_force[lift_actuator_id]:.6f} N')
        print(f'升降执行器力范围:{model.actuator_forcerange[lift_actuator_id]} N')

        print(f'下降后升降自由度驱动力:{data.qfrc_actuator[lift_dof_id]:.6f} N')
        print(f'下降后升降动力学偏置项:{data.qfrc_bias[lift_dof_id]:.6f} N')
        print(f'下降后升降被动力:{data.qfrc_passive[lift_dof_id]:.6f} N')
        print(f'下降后升降约束力:{data.qfrc_constraint[lift_dof_id]:.6f} N')

        down_lift_after = data.joint('joint_lift').qpos[0]
        down_lift_speed = data.joint('joint_lift').qvel[0]

        down_left_tip_after = data.xpos[left_rubber_tip_id].copy()
        down_right_tip_after = data.xpos[right_rubber_tip_id].copy()
        down_midpoint_after = (down_left_tip_after + down_right_tip_after) / 2

        down_actuator_delta_z = down_midpoint_after[2] - descend_midpoint_start_z
        down_height_error = down_midpoint_after[2] - grasp_midpoint_z

        down_remaining_xy = target_point[:2] - down_midpoint_after[:2]
        down_remaining_distance = np.linalg.norm(down_remaining_xy)

        down_lift_error = down_lift_after - descend_lift_goal
        down_elapsed = data.time - down_start_time

        print(f'下降阶段结束原因:{down_exit_reason}')
        print(f'下降阶段耗时:{down_elapsed}')
        print(f'下降后升降位置:{down_lift_after:.6f}m')
        print(f'下降后升降速度:{down_lift_speed:.6f}m/s')
        print(f'升降关节目标误差:{down_lift_error:.6f}m')
        print(f'下降后指尖中点:{down_midpoint_after}')
        print(f'中点实际高度变化:{down_actuator_delta_z:.6f}m')
        print(f'中点高度误差:{down_height_error:.6f}m')
        print(f'下降后水平偏差:{down_remaining_xy}m')
        print(f'下降后水平距离:{down_remaining_distance:.6f}m')

        if not windows.is_running():
            print('窗口已关闭，不执行夹爪闭合任务')
            return

        down_ready = (
            down_exit_reason == '阶段计时结束'
            and abs(down_height_error) < 0.005
            and down_remaining_distance < 0.005
            and abs(down_lift_speed) < 0.001
        )

        if not down_ready:
            print('下降未结束，不执行闭合任务')
            return

        mujoco.mj_forward(model,data)

        close_position_before = data.joint('joint_gripper_slide').qpos[0]

        close_left_before = data.xpos[left_rubber_tip_id].copy()
        close_right_before = data.xpos[right_rubber_tip_id].copy()
        close_gap_before = np.linalg.norm(
            close_left_before - close_right_before
        )


        if target_color == 'red':
            close_goal = -0.005
        else:
            close_goal = 0.0

        close_in_range = (
            gripper_ctrlrange[0] <= close_goal <= gripper_ctrlrange[1]
        )

        if not close_in_range:
            raise RuntimeError('夹爪闭合超出范围')

        close_duration = 1.0
        close_start_time = data.time

        data.ctrl[gripper_id] = close_goal

        while(
            windows.is_running()
            and data.time - close_start_time < close_duration
        ):
            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            close_exit_reason = '阶段计时结束'
        else:
            close_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)

        close_position_after = data.joint('joint_gripper_slide').qpos[0]

        close_left_after = data.xpos[left_rubber_tip_id].copy()
        close_right_after = data.xpos[right_rubber_tip_id].copy()
        close_gap_after = np.linalg.norm(
            close_left_after - close_right_after
        )

        close_position_delta = close_position_after - close_position_before
        close_gap_delta = close_gap_after - close_gap_before
        close_error = close_position_after - close_goal

        close_force = data.actuator_force[gripper_id]
        close_elapsed = data.time - close_start_time

        print(f'闭合阶段结束原因:{close_exit_reason}')
        print(f'闭合阶段耗时:{close_elapsed:.6f}s')
        print(f'闭合目标:{close_goal:.6f}m')
        print(f'闭合后控制命令:{data.ctrl[gripper_id]:.6f}m')
        print(f'闭合前关节位置:{close_position_before:.6f}m')
        print(f'闭合后关节位置:{close_position_after:.6f}m')
        print(f'实际关节变化量:{close_position_delta:.6f}m')
        print(f'夹爪关节目标误差:{close_error:.6f}m')
        print(f'闭合前指尖参考点间距:{close_gap_before:.6f}m')
        print(f'闭合后指尖参考点间距:{close_gap_after:.6f}m')
        print(f'指尖参考点间距变化:{close_gap_delta:.6f}m')
        print(f'闭合后执行器输出力:{close_force:.6f}N')

        mujoco.mj_forward(model,data)

        left_contact_forces = []
        right_contact_forces = []

        contact_force = np.zeros(6, dtype=np.float64)
        force_threshold = 0.001

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            body1_id = model.geom_bodyid[contact.geom1]
            body2_id = model.geom_bodyid[contact.geom2]

            if body1_id == target_body_id:
                other_body_id = body2_id
            elif body2_id == target_body_id:
                other_body_id = body1_id
            else:
                continue

            if(
                other_body_id != left_rubber_tip_id
                and other_body_id != right_rubber_tip_id
            ):
                continue

            mujoco.mj_contactForce(
                model,
                data,
                contact_index,
                contact_force
            )

            normal_force = float(contact_force[0])

            if other_body_id == left_rubber_tip_id:
                left_contact_forces.append(normal_force)
            elif other_body_id == right_rubber_tip_id:
                right_contact_forces.append(normal_force)

        left_has_contact = (
            max(left_contact_forces, default=0.0) > force_threshold
        )

        right_has_contact = (
            max(right_contact_forces, default=0.0) > force_threshold
        )

        both_fingers_contact = left_has_contact and right_has_contact

        print(f'左指与目标物体的接触记录数:{len(left_contact_forces)}')
        print(f'左指各接触点法向力:{left_contact_forces} N')
        print(f'左指是否存在有效目标接触:{left_has_contact}')

        print(f'右指与目标物体的接触记录数:{len(right_contact_forces)}')
        print(f'右指各接触点法向力:{right_contact_forces} N')
        print(f'右指是否存在有效目标接触:{right_has_contact}')

        print(f'两指是否同时存在有效目标接触:{both_fingers_contact}')

        if not both_fingers_contact:
            print('未满足双侧目标接触，不执行试抬')
            return

        if not windows.is_running():
            print('窗口已关闭，不执行试抬')
            return

        if close_exit_reason != '阶段计时结束':
            print('闭合阶段未正常完成，不执行试抬')
            return

        mujoco.mj_forward(model,data)

        lift_test_joint_start = data.joint('joint_lift').qpos[0]

        lift_test_object_before = data.xpos[target_body_id].copy()

        lift_test_left_before = data.xpos[left_rubber_tip_id].copy()
        lift_test_right_before = data.xpos[right_rubber_tip_id].copy()
        lift_test_midpoint_before = (
            (lift_test_left_before + lift_test_right_before) / 2
        )

        lift_test_delta = 0.05
        lift_test_compensation = 0.0229

        lift_test_goal = lift_test_joint_start + lift_test_delta
        lift_test_command = lift_test_goal + lift_test_compensation

        lift_test_goal_in_range = (
            lift_ctrlrange[0] <= lift_test_goal <= lift_ctrlrange[1]
        )

        lift_test_command_in_range = (
            lift_ctrlrange[0] <= lift_test_command <= lift_ctrlrange[1]
        )

        if not (
            lift_test_goal_in_range
            and lift_test_command_in_range
        ):
            raise RuntimeError('试抬目标超出范围')

        print(f'试抬前升降位置:{lift_test_joint_start:.6f}m')
        print(f'试抬前物体世界坐标:{lift_test_object_before}')
        print(f'试抬前物体坐标形状:{lift_test_object_before.shape}')
        print(f'试抬前指尖中点:{lift_test_midpoint_before}')
        print(f'试抬前指尖中点形状:{lift_test_midpoint_before.shape}')
        print(f'计划试抬增加量:{lift_test_delta:.6f}m')
        print(f'试抬关节目标:{lift_test_goal:.6f}m')
        print(f'试抬控制命令:{lift_test_command:.6f}m')
        print(f'试抬目标是否在范围内:{lift_test_goal_in_range}')
        print(f'试抬命令是否在范围内:{lift_test_command_in_range}')

        if not windows.is_running():
            print('试抬计划未结束，不执行试抬')
            return

        data.ctrl[lift_actuator_id] = lift_test_command

        lift_test_duration = 3.0
        lift_test_start_time = data.time

        while(
            windows.is_running()
            and data.time - lift_test_start_time < lift_test_duration
        ):
            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            lift_test_exit_reason = '阶段计时结束'
        else:
            lift_test_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)

        lift_test_joint_after = data.joint('joint_lift').qpos[0]
        lift_test_joint_speed = data.joint('joint_lift').qvel[0]

        lift_test_object_after = data.xpos[target_body_id].copy()

        lift_test_left_after = data.xpos[left_rubber_tip_id].copy()
        lift_test_right_after = data.xpos[right_rubber_tip_id].copy()
        lift_test_midpoint_after = (
            (lift_test_left_after + lift_test_right_after) / 2
        )

        lift_test_joint_delta = lift_test_joint_after - lift_test_joint_start
        lift_test_joint_error = lift_test_joint_after -lift_test_goal
        lift_test_object_rise = lift_test_object_after[2] - lift_test_object_before[2]
        lift_test_midpoint_rise = lift_test_midpoint_after[2] - lift_test_midpoint_before[2]
        lift_test_elapsed = data.time - lift_test_start_time

        print(f'试抬结束原因:{lift_test_exit_reason}')
        print(f'试抬耗时:{lift_test_elapsed:.6f}s')
        print(f'升降关节实际增加量:{lift_test_joint_delta:.6f}m')
        print(f'升降关节目标误差:{lift_test_joint_error:.6f}m')
        print(f'试抬后升降速度:{lift_test_joint_speed:.6f}m/s')
        print(f'试抬后物体世界坐标:{lift_test_object_after}')
        print(f'物体实际上升量:{lift_test_object_rise:.6f}m')
        print(f'指尖中点实际上升量:{lift_test_midpoint_rise:.6f}m')

        if(
            lift_test_exit_reason != '阶段计时结束'
            or not windows.is_running()
        ):
            print('试抬中断或窗口关闭，不执行保持阶段')
            return

        hold_duration = 1.0
        hold_start_time = data.time

        while(
            windows.is_running()
            and data.time - hold_start_time < hold_duration 
        ):
            mujoco.mj_step(model,data)
            windows.sync()
            time.sleep(model.opt.timestep)

        if windows.is_running():
            hold_exit_reason = '阶段计时结束'
        else:
            hold_exit_reason = '窗口提前关闭'

        mujoco.mj_forward(model,data)

        hold_object_after = data.xpos[target_body_id].copy()

        hold_drop = lift_test_object_after[2] - hold_object_after[2]
        hold_object_rise = hold_object_after[2] - lift_test_object_before[2]
        hold_elapsed = data.time - hold_start_time

        print(f'保持阶段结束原因:{hold_exit_reason}')
        print(f'保持阶段耗时:{hold_elapsed:.6f}s')
        print(f'保持后物体世界坐标:{hold_object_after}')
        print(f'保持前后物体下降量:{hold_drop:.6f}m')
        print(f'保持后物体相对初始上升量:{hold_object_rise:.6f}m')

        hold_left_contact_forces = []
        hold_right_contact_forces = []

        hold_contact_force = np.zeros(6, dtype=np.float64)

        for hold_contact_index in range(data.ncon):
            hold_contact = data.contact[hold_contact_index]

            hold_body1_id = model.geom_bodyid[hold_contact.geom1]
            hold_body2_id = model.geom_bodyid[hold_contact.geom2]

            if hold_body1_id == target_body_id:
                other_hold_body_id = hold_body2_id
            elif hold_body2_id == target_body_id:
                other_hold_body_id = hold_body1_id
            else:
                continue

            if(
                other_hold_body_id != left_rubber_tip_id
                and other_hold_body_id != right_rubber_tip_id
            ):
                continue

            mujoco.mj_contactForce(
                model,
                data,
                hold_contact_index,
                hold_contact_force
            )

            normal_hold_force = float(hold_contact_force[0])

            if other_hold_body_id == left_rubber_tip_id:
                hold_left_contact_forces.append(normal_hold_force)
            elif other_hold_body_id == right_rubber_tip_id:
                hold_right_contact_forces.append(normal_hold_force)

        hold_left_has_contact = (
            max(hold_left_contact_forces, default=0.0) > force_threshold
        )

        hold_right_has_contact = (
            max(hold_right_contact_forces, default=0.0) > force_threshold
        )

        hold_both_fingers_contact = (
            hold_left_has_contact
            and hold_right_has_contact
        )

        lift_test_passed = (
            lift_test_exit_reason == '阶段计时结束'
            and hold_exit_reason == '阶段计时结束'
            and hold_object_rise >= 0.03
            and hold_drop <= 0.005
            and hold_both_fingers_contact
        )

        print(f'保持后左指各接触点法向力:{hold_left_contact_forces} N')
        print(f'保持后右指各接触点法向力:{hold_right_contact_forces} N')
        print(f'保持后左指有效接触:{hold_left_has_contact}')
        print(f'保持后右指有效接触:{hold_right_has_contact}')
        print(f'保持后双侧有效接触:{hold_both_fingers_contact}')
        print(f'本次试抬与保持是否通过:{lift_test_passed}')

    print(f'关闭窗口时间:{data.time}')


if __name__ == '__main__':
    main()
