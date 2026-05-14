OMF: otter-motion-format
========

## 简介

用来规范化机器人动作数据格式，方便用于数据采集、训练、日志记录等场合。


## 数据格式


机器人数据一般包含了两大类数据：
1. 目标：一般用于数据采集和训练的目标数据，比如期望关节到达的角度，帧率等。
2. 实际：实际的传感信息，比如电机实际的角度，温度，帧率等。

数据一般又包含不限于：
* 机器人信息，比如型号/关节数量/关节名称
* 世界坐标系下的位姿（位移和旋转）
* 每个关节的旋转角度/速度/加速度/力矩/温度
* 每个部位的位姿/线速度/角速度
* 传感器数据，比如 IMU 姿态，甚至图像
* 数据帧率/运行耗时


基于这些需求，我们定义数据格式：

顶层：

```yaml
version: 1    # 协议版本
format: omf   # 格式标识，方便验证是否是 OMF 格式
basic:        # 基本信息和共用信息，比如数据名称，机器人信息等
target:       # 期望的动作数据
actual:       # 实际的动作记录，用于 log
```

`basic`:

```yaml
version: 1
format: omf
basic:
    name: walk stright
    robot: agibot_x2 # 建议 厂商+型号
    date: 2026-05-13 00:00:00
    joint_names: ["left_hip_pitch", ..., "waist_yaw"] # 关节名，后面的数据严格对应这个顺序
    joint_dims: [1, ...., 1]
    link_names: ["left_ankle_link"]
    imu_names: ["base", "torso"]
    time_names: ["dt", "model_run"]
    data_names: ["target", "model_target", "actual"]
```

`target/model_target/actual`数据，根据`data_names`可以加多种数据：

```yaml
target/model_target/actual:
    fps: 50
    length: 600 # 数据总长
    root_pos: [[0.0, 0.0, 0.0], ...] # root 节点世界坐标系下的位移，单位为米
    root_rot: [[w, x, y, z], ...]    # root 节点世界坐标系下的旋转，四元数
    joint:
        pos:
            - [1.0, ...] # 每个关节旋转的角度, 每个关节的值个数和 dof_dim 对应
            - [1.1, ...] # 对应了 length 个数组
        vel: [] # 类 dof_pos， 关节速度
        acc: [] # 类 dof_pos， 关节加速度
        tau: [] # 类 dof_pos， 关节力矩
    link:
        pos: []     # shape: (length, link_num, 3(xyz)) 世界坐标系下的位移
        rot: []     # shape: (length, link_num, 4(wxyz)) 世界坐标系下的旋转
        lin_vel: [] # shape: (length, link_num, 3(xyz)) 世界坐标系下的线速度
        ang_vel: [] # shape: (length, link_num, 3(wx,wy,wz)) 世界坐标系下的角速度
    imu:
        pos: []  # shape: (length, link_num, 3(xyz)) 世界坐标系下的位移
        rot: []  # shape: (length, link_num, 4(wxyz)) 世界坐标系下的旋转
        gyro: [] # shape: (length, link_num, 3(xyz)) 陀螺仪角速度
        acc: []  # shape: (length, link_num, 3(xyz)) 加速度
        lin_vel: [] # shape: (length, link_num, 3(xyz)) 线速度，可选
    time: [] # shape: (len(time_names), length) 一般用来记录运行时的相关耗时
```


## 保存格式

数据保存一般来说有几种方式：
* json/yaml 文本保存，方便易读但是文件比较大
* pickle 直接保存 Python 变量，方便但是容易库版本不兼容，不利于长期保存
* messagepack 二进制版本的 json,文件更小，只是没法直接文本阅读，需要借用工具

最后选择主要使用 `messagepack` 进行保存，体积小读写速度快，支持的软件也比较多, SDK 支持的语言也很多。
也支持 yaml 和 json， 后缀都是`.msgpack / .yaml / .json`。


## 本库支持的操作

本库是一个 Python 库，有几种可选安装方法：
* `pip install otter-motion-format`：核心读写依赖，不包括显示界面依赖的库
* `pip install "otter-motion-format[viz]"`：加上曲线可视化依赖


支持读取、可视化、保存数据：

```python
import otter_motion_format as omf

motion = omf.load("walk.msgpack")
motion.summary()
motion.show_chart(["target.joint.pos"]) # 默认会显示所有曲线，可以在界面只勾选想要看的曲线，也可以参数传入想看的曲线， rot 会被转换成 rotvec 和欧拉角

motion.save("walk.yaml")
```

添加数据

```python
import otter_motion_format as omf
import numpy as np

robot_joint_names = ["left_hip_pitch", "right_hip_pitch" ] # 设置你的机器人关节名
motion = omf.OMF(name="walk",
                 robot="agibot_x2",
                 joint_names=robot_joint_names,
                 joint_dims=[1] * len(robot_joint_names),
                 data_names = ["target", "model_target", "actual"]
                )

for i in range(100):
    # 直接操作 dict
    motion.target["joint"]["pos"].append([1.0, 1.0 + i*0.01])
    motion.target["joint"]["vel"].append(np.array([0.1, 0.1]))
motion.save("test_motion.msgpack")
```

## 当前已实现功能

已经完成的能力：

* `OMF(...)` 创建标准数据容器
* `load(...)` 读取 `.msgpack/.yaml/.json`
* `save(...)` 按后缀保存 `.msgpack/.yaml/.json`
* `summary()` 输出概要信息
* `show_chart(...)` 打开交互式曲线查看器

曲线查看器目前支持：

* 多种类型数据同时保存，比如 `target` 和 `actual`， 分 tab 查看
* 搜索曲线名
* `All / None / Invert` 快速勾选
* 鼠标拖动平移、滚轮缩放
* 适合大量曲线数据的 pyqtgraph 下采样显示

实现上选择 `pyqtgraph`，原因是：

* 比 `matplotlib` 更适合高频交互和大量点数据
* 依赖比 `echarts` 方案更轻，不需要额外前端打包和 Web 容器
* 后续和 `robot-data-editor` 统一技术栈更顺滑







