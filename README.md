OMF: otter-motion-format
=========

## Overview

OMF standardizes robot motion data for collection, training, playback, logging, and analysis.

Robot motion datasets usually contain at least two categories of data:

1. Target data, such as commanded joint angles and controller timing.
2. Actual data, such as measured joint states, sensor readings, and execution timing.

An OMF file can store multiple named data layers at once, for example `target`, `model_target`, and `actual`.

## Format

Top-level structure:

```yaml
version: 1
format: omf
basic:
target:
actual:
```

The `basic` section contains shared metadata:

```yaml
basic:
  name: walk straight
  robot: agibot_x2
  date: 2026-05-13 00:00:00
  joint_names: ["left_hip_pitch", "right_hip_pitch"]
  joint_dims: [1, 1]
  link_names: ["left_ankle_link"]
  imu_names: ["base", "torso"]
  time_names: ["dt", "model_run"]
  data_names: ["target", "model_target", "actual"]
```

Each named data section listed in `basic.data_names` shares the same layout:

```yaml
target/model_target/actual:
  fps: 50
  length: 600
  root_pos: [[0.0, 0.0, 0.0], ...]
  root_rot: [[w, x, y, z], ...]
  joint:
    pos: []
    vel: []
    acc: []
    tau: []
  link:
    pos: []
    rot: []
    lin_vel: []
    ang_vel: []
  imu:
    pos: []
    rot: []
    gyro: []
    acc: []
    lin_vel: []
  time: []
```

Quaternion storage order in OMF is `wxyz`.

## Storage Backends

Supported file formats:

- `.msgpack` for compact binary storage and fast I/O
- `.yaml` for readable text exports
- `.json` for interoperable text exports

## Installation

Core library:

```bash
pip install otter-motion-format
```

With interactive visualization:

```bash
pip install "otter-motion-format[viz]"
```

## Python Usage

Load and inspect an existing motion:

```python
import otter_motion_format as omf

motion = omf.load("walk.msgpack")
motion.summary()
motion.show_chart(["target.joint.pos"])
motion.save("walk.yaml")
```

Create a new motion container:

```python
import numpy as np
import otter_motion_format as omf

joint_names = ["left_hip_pitch", "right_hip_pitch"]

motion = omf.OMF(
    name="walk",
    robot="agibot_x2",
    joint_names=joint_names,
    joint_dims=[1] * len(joint_names),
    data_names=["target", "model_target", "actual"],
)

for index in range(100):
    motion.target["joint"]["pos"].append([1.0, 1.0 + index * 0.01])
    motion.target["joint"]["vel"].append(np.array([0.1, 0.1]))

motion.save("test_motion.msgpack")
```

`OMF(...)` now requires `name` and `robot` when creating a new container.

## Viewer

`show_chart(...)` opens an interactive viewer built with `PySide6 + pyqtgraph`.

Current viewer behavior:

- Separate tabs for each named data layer
- Searchable channel lists
- `All / None / Invert` quick toggles
- One highlighted current curve per tab
- Checked curves from all tabs drawn together
- Mouse hover crosshair with current time and all visible values
- Drag to pan, wheel to zoom
- Downsampling suitable for dense motion logs

By default, root rotations are expanded into both rotation-vector and Euler-angle channels. You can still force only one representation with `rot_format="rotvec"` or `rot_format="euler"`.

## CLI

View an OMF file:

```bash
otter-motion-format walk.msgpack
otter-motion-format walk.msgpack --rot-format both
```

Convert GMR to OMF:

```bash
otter-gmr-to-omf input.pkl output.msgpack --robot agibot_x2 --data-name target
```

Convert OMF to GMR:

```bash
otter-omf-to-gmr input.msgpack output.pkl --data-name actual
```

## Implemented Capabilities

- `OMF(...)` creates a standard motion container
- `load(...)` reads `.msgpack/.yaml/.json`
- `save(...)` writes `.msgpack/.yaml/.json`
- `summary()` prints a compact overview
- `show_chart(...)` opens the interactive multi-layer curve viewer
- `otter-gmr-to-omf` and `otter-omf-to-gmr` convert between GMR pickle files and OMF

