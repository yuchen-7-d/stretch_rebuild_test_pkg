from pathlib import Path

import mujoco
from head_control import move_head_pan


def main():
    scene_path = Path(
            "/home/yu/stretch_projects/stretch_mujoco/"
            "stretch_mujoco/models/scene.xml"
        )
    
    if not scene_path.is_file():
        raise FileNotFoundError(f'找不到场景文件:{scene_path}')

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    data = mujoco.MjData(model)
    start_time = data.time
    print(f'场景文件:{scene_path}')
    print(f'关节数量:{model.njnt}')
    print(f'执行器数量:{model.nu}')
    print(f'推进前时间:{start_time:.6f}s')

    first_target = 1
    first_result = move_head_pan(model, data, first_target)
    print('第一次动作')
    print(f'目标角度:{first_target:.6f} rad')
    print(f'实际角度:{first_result["actual_angle"]:.6f} rad')
    print(f'最终速度:{first_result["actual_velocity"]:.6f} rad/s')
    print(f'角度误差:{first_result["angle_error"]:.6f} rad')
    print(f'结束原因:{first_result["reason"]}')
    print(f'本次动作耗时:{first_result["elapsed_time"]:.6f}s')

    if first_result["reason"] == '到位并停稳':
        second_target = -1
        second_result = move_head_pan(model, data, second_target)
        print('第二次动作')
        print(f'目标角度:{second_target:.6f} rad')
        print(f'实际角度:{second_result["actual_angle"]:.6f} rad')
        print(f'最终速度:{second_result["actual_velocity"]:.6f} rad/s')
        print(f'角度误差:{second_result["angle_error"]:.6f} rad')
        print(f'结束原因:{second_result["reason"]}')
        print(f'本次动作耗时:{second_result["elapsed_time"]:.6f}s')

    else:
        print('第一次还没成功，不执行第二次')

    print(f'推进后时间:{data.time:.6f}s')



if __name__ == '__main__':
    main()
