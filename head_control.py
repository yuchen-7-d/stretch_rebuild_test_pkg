import mujoco
import mujoco.viewer
import time


def move_head_pan(model, data, target_angle):
    start_time = data.time

    stable_since = None
    exit_reason = '窗口提前关闭'
    position_tolerance = 0.01
    velocity_tolerance = 0.01
    settle_duration = 0.2
    timeout_duration = 5

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

    lower_limit, upper_limit = head_pan_ctrl_range
    if not (lower_limit <= target_angle <=upper_limit):
        raise ValueError(f'目标角度超出控制范围:{target_angle}')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[head_pan_actuator_id] = target_angle
            mujoco.mj_step(model, data)
            viewer.sync()

            joint_state = data.joint('joint_head_pan')
            actual_angle = joint_state.qpos[0]
            actual_velocity = joint_state.qvel[0]
            angle_error = actual_angle - target_angle

            within_tolerance = abs(angle_error) < position_tolerance and abs(actual_velocity) < velocity_tolerance
            if within_tolerance:
                if stable_since is None:
                    stable_since = data.time
                if settle_duration <= data.time - stable_since:
                    exit_reason = '到位并停稳'
                    break
            else:
                stable_since = None
            if data.time - start_time >= timeout_duration:
                exit_reason = '超时'
                break

            time.sleep(model.opt.timestep)

    joint_state = data.joint('joint_head_pan')
    actual_angle = joint_state.qpos[0]
    actual_velocity = joint_state.qvel[0]
    angle_error = actual_angle - target_angle

    return{
        'reason': exit_reason,
        'elapsed_time': data.time - start_time,
        'actual_angle': actual_angle,
        'actual_velocity': actual_velocity,
        'angle_error': angle_error
    }