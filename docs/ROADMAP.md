# Roadmap

Goal: a desktop application (C++ / Qt 6) to operate the SO-101 and run learned skills safely.
Reference task: pick a blue cube from the table and hand it to a person.

## Target architecture

| Component | Now | Next |
|---|---|---|
| UI | PyQt6 prototype | C++ / Qt 6 app: Qt Quick 3D digital twin, task pipeline editor, chat-driven actions |
| Robot runtime | `robotd` (Python, HTTP) | `robotd` with ZeroMQ or gRPC; ROS 2 `ros2_control` hardware interface in C++ |
| Kinematics | numpy FK/IK from URDF | same algorithms in C++ (or KDL / Pinocchio), MoveIt 2 for planning |
| Simulation | first-order servo model | MuJoCo twin: mirror (real to sim) and preview (sim to real after confirmation) |
| Policies | none | GPU policy server: ACT, SmolVLA, Diffusion Policy via LeRobot async inference |

Inter-process channels:

| Channel | Rate | Content |
|---|---|---|
| `state` (PUB/SUB) | 30-50 Hz | `t, q[6], q_goal[6], load[6], temp[6], torque_on` |
| `command` (REQ/REP) | on event | jog, goto, gripper, hold, torque on/off; always through the safety filter |
| `video` | 30 Hz | frames from two cameras (shared memory or JPEG) |
| `policy` (gRPC) | 10-30 Hz | observation to action chunk |

The safety filter sits in front of the driver for every command source: URDF and calibration limits, per-tick step
limit, speed limit, self-collision check with sphere models, watchdog on link loss.

## Stages

| # | Stage | Done when |
|---|---|---|
| 0 | Manual control prototype: joint and Cartesian jog, gripper, 2 cameras, 3D skeleton, diagnostics, sim mode | works without the robot; with the robot: 6/6 motors answer, smooth motion on command (**done**) |
| 1 | Model-to-hardware binding: joint signs and zeros, camera calibration, front camera pose | 3D skeleton matches the robot within 3 deg on 5 poses (signs and zeros **done**) |
| 2 | MuJoCo digital twin: MJCF, table, cube; mirror and preview modes | mirror latency under 50 ms |
| 3 | Data: LeRobotDataset episodes (2 cameras, state, action) from teleoperation | 50+ episodes, replay reproduces them |
| 4 | Policies: train ACT and SmolVLA, policy server, success check from the camera | measured success rate on 20 trials |
| 5 | Runtime as a service: `robotd` over ZeroMQ or gRPC, ROS 2 `ros2_control` interface | UI restarts without reconnecting the robot |
| 6 | C++ / Qt 6 application with the full flow | the whole scenario runs without a terminal |

## Pipeline data model

`Pipeline` = ordered list of `Action`.

`Action`: `name`, `instruction` (text), `executor` (`policy` | `planner` | `script`), `checkpoint`, `dataset`,
`params`, `success_predicate`, `timeout_s`, `on_fail` (`retry` | `stop`).

Execution: `Action` -> executor -> stream of action chunks -> safety filter -> robot; state -> 3D view and MuJoCo;
success predicate -> next action, retry or stop.
