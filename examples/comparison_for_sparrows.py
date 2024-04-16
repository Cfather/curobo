#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library

# Third Party
import torch
import scipy.io as sio
import numpy as np

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def plot_traj(trajectory, dt):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(4, 1)
    q = trajectory.position.cpu().numpy()
    qd = trajectory.velocity.cpu().numpy()
    qdd = trajectory.acceleration.cpu().numpy()
    qddd = trajectory.jerk.cpu().numpy()
    timesteps = [i * dt for i in range(q.shape[0])]
    for i in range(q.shape[-1]):
        axs[0].plot(timesteps, q[:, i], label=str(i))
        axs[1].plot(timesteps, qd[:, i], label=str(i))
        axs[2].plot(timesteps, qdd[:, i], label=str(i))
        axs[3].plot(timesteps, qddd[:, i], label=str(i))

    plt.legend()
    plt.savefig("test.png")
    plt.close()
    # plt.show()


def plot_iters_traj(trajectory, d_id=1, dof=7, seed=0):
    # Third Party
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(len(trajectory), 1)
    if len(trajectory) == 1:
        axs = [axs]
    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            axs[k].plot(
                q[i][seed, :-1, d_id].cpu(),
                "r+-",
                label=str(i),
                alpha=0.1 + min(0.9, float(i) / (len(q))),
            )
    plt.legend()
    plt.show()


def plot_iters_traj_3d(trajectory, d_id=1, dof=7, seed=0):
    # Third Party
    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")
    c = 0
    h = trajectory[0][0].shape[1] - 1
    x = [x for x in range(h)]

    for k in range(len(trajectory)):
        q = trajectory[k]

        for i in range(len(q)):
            # ax.plot3D(x,[c for _ in range(h)],  q[i][seed, :, d_id].cpu())#, 'r')
            ax.scatter3D(
                x, [c for _ in range(h)], q[i][seed, :h, d_id].cpu(), c=q[i][seed, :, d_id].cpu()
            )
            # @plt.show()
            c += 1
    # plt.legend()
    plt.show()

def demo_motion_gen():
    # Standard Library
    tensor_args = TensorDeviceType()
    world_file = "simple_scenario.yml"
    robot_file = "kinova_gen3.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_dt=0.01,
        # trajopt_dt=0.15,
        # velocity_scale=0.1,
        use_cuda_graph=True,
        # finetune_dt_scale=2.5,
        interpolation_steps=10000,
    )

    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(parallel_finetune=True)

    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=js, parallel_finetune=True)
    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    # robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    start_state_tensor = torch.tensor([0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.345], device='cuda:0')
    goal_state_tensor = torch.tensor([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, -1.345], device='cuda:0')

    start_state = JointState.from_position(start_state_tensor.view(1, -1))
    goal_state = JointState.from_position(goal_state_tensor.view(1, -1))

    result = motion_gen.plan_single_js(
        start_state,
        goal_state,
        MotionGenPlanConfig(max_attempts=1, parallel_finetune=True, enable_finetune_trajopt=True),
    )
    # print(
    #     "Trajectory Generated: ",
    #     result.success,
    #     result.solve_time,
    #     result.status,
    #     result.optimized_dt,
    # )

    if_success = result.success.cpu().numpy()
    traj = result.get_interpolated_plan()
    q = traj.position.cpu().numpy()
    dt = result.optimized_dt.cpu().numpy()
    solve_time = result.solve_time

    print(if_success)

    # save q as a mat file
    sio.savemat('curobo_trajectory.mat', {'if_success': if_success, 'q': q, 'dt': dt, 'solve_time': solve_time})

if __name__ == "__main__":
    setup_curobo_logger("error")
    demo_motion_gen()
