OMF: otter-motion-format
========

[中文](./README_ZH.md) | English

## Introduction

A standardized format for robot motion data, designed for data collection, training, logging, and related applications.
It helps unify data representation and supports conversion between common file formats.


## Data Format


Robot data generally contains two major categories:
1. Target: Target data used for data collection and training, such as desired joint angles and frame rates.
2. Actual: Actual sensor data, such as measured motor angles, temperatures, and frame rates.

The data may include, but is not limited to:

* Robot information, such as model / number of joints / joint names
* Pose in the world coordinate system (translation and rotation)
* Joint angle / velocity / acceleration / torque / temperature for each joint
* Pose / linear velocity / angular velocity for each link
* Sensor data, such as IMU orientation and even images
* Data frame rate / execution time


Based on these requirements, we define the following format:

Top level:

```yaml
version: 1    # Protocol version
format: omf   # Format identifier, used to verify that this is an OMF file
basic:        # Basic and shared information, such as data name and robot information
target:       # Desired motion data
actual:       # Actual recorded motion data, used for logs
````

`basic`:

```yaml
version: 1
format: omf
basic:
    name: walk stright
    robot: agibot_x2 # Recommended format: manufacturer + model
    date: 2026-05-13 00:00:00
    joint_names: ["left_hip_pitch", ..., "waist_yaw"] # Joint names; all subsequent data strictly follows this order
    joint_dims: [1, ...., 1]
    link_names: ["left_ankle_link"]
    imu_names: ["base", "torso"]
    time_names: ["dt", "model_run"]
    data_names: ["target", "model_target", "actual"]
```

`target/model_target/actual` data can contain multiple data groups according to `data_names`:

```yaml
target/model_target/actual:
    fps: 50
    length: 600 # Total number of frames
    root_pos: [[0.0, 0.0, 0.0], ...] # Root position in the world coordinate system, in meters
    root_rot: [[w, x, y, z], ...]    # Root rotation in the world coordinate system, quaternion
    joint:
        pos:
            - [1.0, ...] # Joint angles; the number of values for each joint corresponds to dof_dim
            - [1.1, ...] # Contains `length` arrays
        vel: [] # Same format as joint.pos, joint velocities
        acc: [] # Same format as joint.pos, joint accelerations
        tau: [] # Same format as joint.pos, joint torques
    link:
        pos: []     # shape: (length, link_num, 3(xyz)) Position in the world coordinate system
        rot: []     # shape: (length, link_num, 4(wxyz)) Rotation in the world coordinate system
        lin_vel: [] # shape: (length, link_num, 3(xyz)) Linear velocity in the world coordinate system
        ang_vel: [] # shape: (length, link_num, 3(wx,wy,wz)) Angular velocity in the world coordinate system
    imu:
        pos: []     # shape: (length, link_num, 3(xyz)) Position in the world coordinate system
        rot: []     # shape: (length, link_num, 4(wxyz)) Rotation in the world coordinate system
        gyro: []    # shape: (length, link_num, 3(xyz)) Gyroscope angular velocity
        acc: []     # shape: (length, link_num, 3(xyz)) Acceleration
        lin_vel: [] # shape: (length, link_num, 3(xyz)) Linear velocity, optional
    time: [] # shape: (len(time_names), length); typically used to record execution timings
```

## Storage Formats

There are several common ways to store data:

* JSON/YAML text files: easy to read, but relatively large in size
* Pickle: directly stores Python variables, convenient but may be incompatible across library versions and is not suitable for long-term storage
* MessagePack: a binary version of JSON, producing smaller files; not directly human-readable but can be inspected using tools

The primary storage format chosen is `MessagePack`, which offers small file size, fast read/write performance, and broad software support across many programming languages.
YAML and JSON are also supported, with file extensions `.msgpack`, `.yaml`, and `.json`.

## Supported Operations

This library is a Python package and can be installed in several ways:

* `pip install otter-motion-format`: core read/write functionality only, without visualization dependencies
* `pip install "otter-motion-format[viz]"`: includes curve visualization dependencies
* `pip install "otter-motion-format[all]"`: installs all dependencies; currently the same as `viz`

Supports loading, visualization, and saving:

```python
import otter_motion_format as omf

motion = omf.load("walk.msgpack")
motion.summary()                        # Print a summary of the data in the terminal
motion.show_chart(["target.joint.pos"]) # By default, all curves are shown. You can select curves in the GUI or pass specific curve names.
                                        # Rotations are automatically converted to rotation vectors and Euler angles.

motion.save("walk.yaml")
```

Adding data:

```python
import otter_motion_format as omf
import numpy as np

robot_joint_names = ["left_hip_pitch", "right_hip_pitch"] # Define your robot joint names
motion = omf.OMF(name="walk",
                 robot="agibot_x2",
                 joint_names=robot_joint_names,
                 joint_dims=[1] * len(robot_joint_names),
                 data_names=["target", "model_target", "actual"]
                )

for i in range(100):
    # Directly manipulate the dictionary
    motion.target["joint"]["pos"].append([1.0, 1.0 + i * 0.01])
    motion.target["joint"]["vel"].append(np.array([0.1, 0.1]))
motion.save("test_motion.msgpack")
```

## Visualization Features

As mentioned earlier, `motion.show_chart()` can be used to display curves in a GUI. You can also open files directly from the command line:

```shell
otter-motion-format "path/to/data_file.msgpack"
```

Note that you need to install the visualization dependencies first using
`pip install "otter-motion-format[viz]"` or `pip install "otter-motion-format[all]"`.

The curve viewer currently supports:

* Saving multiple data types simultaneously, such as `target` and `actual`, with separate tabs for each
* Searching for curve names
* `All / None / Invert` quick selection
* Left mouse button to pan, right mouse button to zoom quickly, mouse wheel to zoom
* Viewing the value at the current cursor position
* Customizing curve colors


## LICENSE

[Apache License 2.0](./LICENSE)

In short: Feel free to use this project and share suggestions for improvement. Commercial use is allowed and please keep LICENSE file. If you fork and modify it, please document your changes. Pull requests are welcome.
