from pathlib import Path

import mujoco
import cv2 as cv
import numpy as np
import math


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

    head_pan_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'head_pan'
    )

    head_tilt_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        'head_tilt'
    )

    head_ids = {"head_pan":head_pan_id, "head_tilt":head_tilt_id}
    for head_name, head_id in head_ids.items():
        if head_id == -1:
            raise RuntimeError(f'找不到Mj_name:{head_name}')

    data.ctrl[head_pan_id] = -1.57
    data.ctrl[head_tilt_id] = -0.9

    mujoco.mj_forward(model, data)

    print(f'推进前时间:{data.time:.6f}')
    print(f'推进前蓝色中心高度:{data.body("object1").xpos[2]:.6f}m')
    print(f'推进前红色中心高度:{data.body("object2").xpos[2]:.6f}m')

    run_duration = 1.0
    step_count = round(run_duration / model.opt.timestep)

    print(f'本次推进步数:{step_count}')

    for _ in range(step_count):
        mujoco.mj_step(model,data)

    mujoco.mj_forward(model, data)

    print(f'推进后时间:{data.time:.6f}')
    print(f'推进后蓝色中心高度:{data.body("object1").xpos[2]:.6f}m')
    print(f'推进后红色中心高度:{data.body("object2").xpos[2]:.6f}m')

    print(f'头部实际水平角度:{data.joint("joint_head_pan").qpos[0]:.6f}rad')
    print(f'头部实际俯仰角度:{data.joint("joint_head_tilt").qpos[0]:.6f}rad')

    object1_body = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'object1'
    )

    object2_body = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        'object2'
    )

    object_bodys = {'object1':object1_body, 'object2':object2_body}
    for object_name, object_body in object_bodys.items():
        if object_body == -1:
            raise RuntimeError(f'找不到Mj_name:{object_name}')

    blue_velocity = np.zeros((6,),dtype=np.float64)
    red_velocity = np.zeros((6,),dtype=np.float64)

    linear_speed_tol = 0.001
    angular_speed_tol = 0.01

    start_time = data.time
    stable_since = None
    settle_duration = 0.2
    timeout_duration = 2.0
    settled = False

    while data.time - start_time < timeout_duration:
        mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            object1_body,
            blue_velocity,
            0
        )
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            object2_body,
            red_velocity,
            0
        )

        blue_angular_speed = np.linalg.norm(blue_velocity[:3])
        blue_linear_speed = np.linalg.norm(blue_velocity[3:])

        red_angular_speed = np.linalg.norm(red_velocity[:3])
        red_linear_speed = np.linalg.norm(red_velocity[3:])

        blue_slow = (
            blue_angular_speed < angular_speed_tol
            and blue_linear_speed < linear_speed_tol 
        )

        red_slow = (
            red_angular_speed < angular_speed_tol
            and red_linear_speed < linear_speed_tol 
        )

        both_slow = red_slow and blue_slow

        if both_slow:
            if stable_since is None:
                stable_since = data.time

            if data.time - stable_since >= settle_duration:
                settled = True
                break
        else:
            stable_since = None

    print(f'停稳检查是否通过:{settled}')
    print(f'本次等待耗时:{data.time - start_time:.6f}s')

    print(f'蓝色角速度大小:{blue_angular_speed:.6e} rad/s')
    print(f'蓝色线速度大小:{blue_linear_speed:.6e} m/s')
    print(f'红色角速度大小:{red_angular_speed:.6e} rad/s')
    print(f'红色线速度大小:{red_linear_speed:.6e} m/s')
    print(f'两物体本帧是否满足低速条件:{both_slow}')

    print(f'蓝色速度数组:{blue_velocity}')
    print(f'蓝色速度数组形状:{blue_velocity.shape}')
    print(f'红色速度数组:{red_velocity}')
    print(f'红色速度数组形状:{red_velocity.shape}')


    if not settled:
        raise RuntimeError('等待物体停稳时，停止本次采集')

    camera_position = data.cam_xpos[d435i_camera_rgb_id].copy()
    camera_rotation = data.cam_xmat[d435i_camera_rgb_id].reshape(3, 3).copy()
    pose_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_pose.npz')
    np.savez(
        pose_path,
        position= camera_position,
        rotation= camera_rotation
    )

    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.update_scene(data, camera=d435i_camera_rgb_id)
        rgb_image = renderer.render()
        renderer.enable_depth_rendering()
        depth_image = renderer.render()

    rgb_image_height = rgb_image.shape[0]
    rgb_image_width = rgb_image.shape[1]
    fovy_deg = model.cam_fovy[d435i_camera_rgb_id]
    fovy_rad = np.deg2rad(fovy_deg)

    fy = rgb_image_height / (2 * math.tan (fovy_rad/2))
    fx = fy
    cx = (rgb_image_width - 1) / 2
    cy = (rgb_image_height - 1) / 2

    intrinsics_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_intrinsics.npz')
    np.savez(
        intrinsics_path,
        fx= fx,
        fy= fy,
        cx= cx,
        cy= cy,
        height= rgb_image_height,
        width= rgb_image_width
    )

    depth_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_depth.npy')
    np.save(depth_path, depth_image)

    loaded_depth = np.load(depth_path)

    bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

    image_path = Path('/home/yu/stretch_rebuild/saved_picture/camera_rgb.png')
    image_saved = cv.imwrite(str(image_path), bgr_image)

    if not image_saved:
        raise RuntimeError(f'图片保存失败:{image_path}')

    print(f'图像状态:{rgb_image.shape}')
    print(f'图像类型:{rgb_image.dtype}')
    print(f'图像宽度:{rgb_image_width}')
    print(f'图像高度:{rgb_image_height}')
    print(f'垂直视场角:{fovy_deg} deg')
    print(f'fx:{fx:.6f}')
    print(f'fy:{fy:.6f}')
    print(f'cx:{cx:.6f}')
    print(f'cy:{cy:.6f}')
    print(f'相机内参保存至:{intrinsics_path}')

    print(f'相机位置:{camera_position}')
    print(f'相机位置数组:{camera_position.shape}')
    print(f'相机朝向:{camera_rotation}')
    print(f'相机朝向数组:{camera_rotation.shape}')
    print(f'相机位置朝向保存位置:{pose_path}')

    print(f'保存状态:{image_saved}')
    print(f'保存至:{image_path}')
    print(f'深度图形状:{depth_image.shape}')
    print(f'深度图类型:{depth_image.dtype}')
    print(f'最小深度:{depth_image.min():.6f}m')
    print(f'最大深度:{depth_image.max():.6f}m')
    print(f'深度图保存路径:{depth_path}')

    print(f'加载深度图形状:{loaded_depth.shape}')
    print(f'加载深度图类型:{loaded_depth.dtype}')
    print(f'保存前后数组是否一致:{np.array_equal(loaded_depth, depth_image)}')


if __name__ == '__main__':
    main()
