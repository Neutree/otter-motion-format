from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import yaml


def _deep_convert_for_storage(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		return {key: _deep_convert_for_storage(sub_value) for key, sub_value in value.items()}
	if isinstance(value, (list, tuple)):
		return [_deep_convert_for_storage(item) for item in value]
	return value


def _deep_copy_data(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return value.copy()
	if isinstance(value, dict):
		return {key: _deep_copy_data(sub_value) for key, sub_value in value.items()}
	if isinstance(value, list):
		return [_deep_copy_data(item) for item in value]
	return deepcopy(value)


def _as_float_array(values: Any, width: int | None = None) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	if array.size == 0:
		if width is None:
			return np.zeros((0,), dtype=np.float64)
		return np.zeros((0, width), dtype=np.float64)
	if width is not None:
		array = np.asarray(array, dtype=np.float64).reshape((-1, width))
	return np.asarray(array, dtype=np.float64)


def _normalize_time_array(values: Any, *, time_name_count: int = 0) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	if array.size == 0:
		return np.zeros((0, max(int(time_name_count), 0)), dtype=np.float64)
	if array.ndim == 1:
		if time_name_count not in (0, 1):
			raise ValueError(f"time must have {time_name_count} columns, got 1")
		return array.reshape((-1, 1))
	if array.ndim != 2:
		raise ValueError(f"time must be a 1D or 2D array, got shape {array.shape}")
	if time_name_count > 0 and array.shape[1] != time_name_count:
		raise ValueError(f"time must have {time_name_count} columns, got {array.shape[1]}")
	return array


def _quaternion_wxyz_to_euler_xyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
	w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
	sinr_cosp = 2.0 * (w * x + y * z)
	cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2.0 * (w * y - z * x)
	pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)

	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)
	return np.array([roll, pitch, yaw], dtype=np.float64)


def _quaternion_wxyz_to_rotvec(quaternion_wxyz: np.ndarray) -> np.ndarray:
	quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
	norm = np.linalg.norm(quaternion_wxyz)
	if norm <= 1e-12:
		return np.zeros(3, dtype=np.float64)
	quaternion_wxyz = quaternion_wxyz / norm
	w = float(np.clip(quaternion_wxyz[0], -1.0, 1.0))
	xyz = quaternion_wxyz[1:]
	angle = 2.0 * np.arccos(w)
	sin_half = np.sqrt(max(0.0, 1.0 - w * w))
	if sin_half <= 1e-12:
		return np.zeros(3, dtype=np.float64)
	axis = xyz / sin_half
	return axis * angle


class OMF:
	VERSION = 1
	FORMAT = "omf"
	DEFAULT_DATA_NAMES = ["target", "actual"]
	_RESERVED_TOP_LEVEL_KEYS = {"version", "format", "basic"}

	def __init__(
		self,
		name: str | None = None,
		robot: str | None = None,
		joint_names: list[str] | None = None,
		joint_dims: list[int] | None = None,
		link_names: list[str] | None = None,
		imu_names: list[str] | None = None,
		time_names: list[str] | None = None,
		data_names: list[str] | None = None,
		date: str | None = None,
		data: dict[str, Any] | None = None,
	) -> None:
		if data is None:
			resolved_name = str(name or "").strip()
			resolved_robot = str(robot or "").strip()
			if not resolved_name:
				raise ValueError("name is required when creating a new OMF")
			if not resolved_robot:
				raise ValueError("robot is required when creating a new OMF")
			joint_names = list(joint_names or [])
			joint_dims = list(joint_dims or ([1] * len(joint_names)))
			if joint_names and len(joint_names) != len(joint_dims):
				raise ValueError("joint_names and joint_dims must have the same length")
			resolved_data_names = self._normalize_data_names(data_names or self.DEFAULT_DATA_NAMES)
			self.data = {
				"version": self.VERSION,
				"format": self.FORMAT,
				"basic": {
					"name": resolved_name,
					"robot": resolved_robot,
					"date": date,
					"joint_names": joint_names,
					"joint_dims": joint_dims,
					"link_names": list(link_names or []),
					"imu_names": list(imu_names or []),
					"time_names": list(time_names or []),
					"data_names": resolved_data_names,
				},
			}
			for data_name in resolved_data_names:
				self.data[data_name] = self._empty_section()
		else:
			self.data = _deep_copy_data(data)
		self._ensure_defaults()
		self.validate()

	@staticmethod
	def _empty_section() -> dict[str, Any]:
		return {
			"fps": 0,
			"length": 0,
			"root_pos": [],
			"root_rot": [],
			"joint": {
				"pos": [],
				"vel": [],
				"acc": [],
				"tau": [],
			},
			"link": {
				"pos": [],
				"rot": [],
				"lin_vel": [],
				"ang_vel": [],
			},
			"imu": {
				"pos": [],
				"rot": [],
				"gyro": [],
				"acc": [],
				"lin_vel": [],
			},
			"time": [],
		}

	@classmethod
	def _normalize_data_names(cls, data_names: list[str]) -> list[str]:
		resolved: list[str] = []
		seen: set[str] = set()
		for data_name in data_names:
			name = str(data_name).strip()
			if not name:
				continue
			if name in cls._RESERVED_TOP_LEVEL_KEYS:
				raise ValueError(f"data name '{name}' is reserved")
			if name not in seen:
				seen.add(name)
				resolved.append(name)
		if not resolved:
			return list(cls.DEFAULT_DATA_NAMES)
		return resolved

	def _infer_data_names_from_payload(self) -> list[str]:
		basic = self.data.setdefault("basic", {})
		data_names = basic.get("data_names")
		if isinstance(data_names, list) and data_names:
			return self._normalize_data_names(data_names)
		inferred = [key for key in self.data.keys() if key not in self._RESERVED_TOP_LEVEL_KEYS and isinstance(self.data.get(key), dict)]
		return self._normalize_data_names(inferred or self.DEFAULT_DATA_NAMES)

	def _ensure_defaults(self) -> None:
		self.data.setdefault("version", self.VERSION)
		self.data.setdefault("format", self.FORMAT)
		basic = self.data.setdefault("basic", {})
		basic.setdefault("name", "")
		basic.setdefault("robot", "")
		basic.setdefault("date", None)
		basic.setdefault("joint_names", [])
		basic.setdefault("joint_dims", [1] * len(basic["joint_names"]))
		basic.setdefault("link_names", [])
		basic.setdefault("imu_names", [])
		basic.setdefault("time_names", [])
		basic["data_names"] = self._infer_data_names_from_payload()
		for data_name in basic["data_names"]:
			section = self.data.setdefault(data_name, self._empty_section())
			default = self._empty_section()
			for key, value in default.items():
				section.setdefault(key, _deep_copy_data(value))

	@property
	def basic(self) -> dict[str, Any]:
		return self.data["basic"]

	@property
	def data_names(self) -> list[str]:
		return list(self.basic.get("data_names", []))

	@property
	def target(self) -> dict[str, Any]:
		return self.data.setdefault("target", self._empty_section())

	@property
	def actual(self) -> dict[str, Any]:
		return self.data.setdefault("actual", self._empty_section())

	@property
	def name(self) -> str:
		return str(self.basic.get("name", ""))

	def __getattr__(self, name: str) -> Any:
		if name in self.data_names:
			return self.data[name]
		raise AttributeError(name)

	def data_section(self, data_name: str) -> dict[str, Any]:
		if data_name not in self.data_names:
			raise KeyError(data_name)
		return self.data[data_name]

	def iter_data_sections(self) -> list[tuple[str, dict[str, Any]]]:
		return [(data_name, self.data[data_name]) for data_name in self.data_names]

	def clone(self) -> "OMF":
		return OMF(data=_deep_copy_data(self.data))

	def validate(self) -> None:
		if self.data.get("format") != self.FORMAT:
			raise ValueError(f"Unsupported format: {self.data.get('format')}")
		if int(self.data.get("version", 0)) != self.VERSION:
			raise ValueError(f"Unsupported version: {self.data.get('version')}")

		joint_names = list(self.basic.get("joint_names", []))
		joint_dims = list(self.basic.get("joint_dims", []))
		if len(joint_names) != len(joint_dims):
			raise ValueError("basic.joint_names and basic.joint_dims must have the same length")
		if any(int(dim) <= 0 for dim in joint_dims):
			raise ValueError("basic.joint_dims must contain positive integers")
		self.basic["data_names"] = self._normalize_data_names(self.data_names)

		for data_name in self.data_names:
			section = self.data.setdefault(data_name, self._empty_section())
			self._normalize_section(section)
			self._validate_section(data_name, section)

	def _normalize_section(self, section: dict[str, Any]) -> None:
		time_values = _normalize_time_array(
			section.get("time", []),
			time_name_count=len(self.basic.get("time_names", [])),
		)
		section["time"] = time_values.tolist() if time_values.size > 0 else []
		section["fps"] = int(section.get("fps", 0) or 0)
		computed_length = self._section_length(section)
		section["length"] = max(int(section.get("length", 0) or 0), computed_length)

	def _section_length(self, section: dict[str, Any]) -> int:
		candidates = []
		for key in ("root_pos", "root_rot"):
			value = section.get(key, [])
			if isinstance(value, (list, np.ndarray)):
				candidates.append(len(value))
		time_values = _normalize_time_array(
			section.get("time", []),
			time_name_count=len(self.basic.get("time_names", [])),
		)
		if time_values.size > 0:
			candidates.append(time_values.shape[0])
		for group_name in ("joint", "link", "imu"):
			group = section.get(group_name, {})
			for value in group.values():
				if isinstance(value, (list, np.ndarray)):
					candidates.append(len(value))
		return max(candidates, default=0)

	def _validate_section(self, data_name: str, section: dict[str, Any]) -> None:
		length = int(section.get("length", 0))
		if length < 0:
			raise ValueError(f"{data_name}.length must be >= 0")
		for key in ("root_pos", "root_rot", "time"):
			value = section.get(key, [])
			if len(value) not in (0, length):
				raise ValueError(f"{data_name}.{key} length mismatch: expected 0 or {length}, got {len(value)}")
		time_values = _normalize_time_array(section.get("time", []), time_name_count=len(self.basic.get("time_names", [])))
		if time_values.shape[0] not in (0, length):
			raise ValueError(f"{data_name}.time length mismatch: expected 0 or {length}, got {time_values.shape[0]}")
		for group_name, key_specs in {
			"joint": ("pos", "vel", "acc", "tau"),
			"link": ("pos", "rot", "lin_vel", "ang_vel"),
			"imu": ("pos", "rot", "gyro", "acc", "lin_vel"),
		}.items():
			group = section.get(group_name, {})
			for key in key_specs:
				value = group.get(key, [])
				if len(value) not in (0, length):
					raise ValueError(
						f"{data_name}.{group_name}.{key} length mismatch: expected 0 or {length}, got {len(value)}"
					)

	def summary(self) -> str:
		self.validate()
		lines = [
			f"OMF(name={self.name!r}, robot={self.basic.get('robot', '')!r})",
			f"  joints: {len(self.basic.get('joint_names', []))}",
			f"  links: {len(self.basic.get('link_names', []))}",
			f"  imus: {len(self.basic.get('imu_names', []))}",
			f"  data_names: {self.data_names}",
		]
		for data_name, section in self.iter_data_sections():
			lines.append(f"  {data_name}: fps={section.get('fps', 0)}, length={section.get('length', 0)}")
		text = "\n".join(lines)
		print(text)
		return text

	def to_dict(self) -> dict[str, Any]:
		self.validate()
		return _deep_copy_data(self.data)

	def save(self, path: str | Path) -> None:
		self.validate()
		path = Path(path)
		payload = _deep_convert_for_storage(self.data)
		suffix = path.suffix.lower()
		if suffix == ".msgpack":
			packed = msgpack.packb(payload, use_bin_type=True)
			path.write_bytes(packed)
			return
		if suffix in {".yaml", ".yml"}:
			path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
			return
		if suffix == ".json":
			path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
			return
		raise ValueError(f"Unsupported output format: {suffix}")

	def show_chart(self, keys: list[str] | None = None, rot_format: str | None = None) -> None:
		self.validate()
		try:
			from .viewer import ChannelSpec, show_omf_viewer
		except ImportError as exc:
			raise ImportError(
				"Visualization requires optional dependencies. Install with: pip install 'otter-motion-format[viz]'"
			) from exc

		sections: dict[str, list[ChannelSpec]] = {}
		for data_name in self.data_names:
			sections[data_name] = [
				ChannelSpec(
					key=channel["key"],
					label=channel["label"],
					values=channel["values"],
					x_values=channel["x_values"],
					layer_name=data_name,
				)
				for channel in self._build_channels(data_name=data_name, rot_format=rot_format)
			]
		preselected = self._resolve_preselected_keys(keys=keys, rot_format=rot_format)
		show_omf_viewer(
			title=self.name or "OMF Viewer",
			sections=sections,
			preselected=preselected,
			layer_styles=self._default_layer_styles(),
		)

	def _default_layer_styles(self) -> dict[str, dict[str, Any]]:
		styles: dict[str, dict[str, Any]] = {}
		fallback_patterns = ["dash", "dashdot", "dot", "dash", "dashdot"]
		for index, data_name in enumerate(self.data_names):
			if data_name == "actual":
				styles[data_name] = {"line_style": "solid", "opacity": 1.0, "width": 2.0}
			elif data_name == "target":
				styles[data_name] = {"line_style": "dash", "opacity": 0.8, "width": 2.0}
			elif data_name == "model_target":
				styles[data_name] = {"line_style": "dashdot", "opacity": 0.9, "width": 2.0}
			else:
				styles[data_name] = {
					"line_style": fallback_patterns[index % len(fallback_patterns)],
					"opacity": max(0.45, 0.85 - 0.08 * index),
					"width": 1.8,
				}
		return styles

	def _resolve_preselected_keys(self, keys: list[str] | None, rot_format: str | None) -> dict[str, set[str]]:
		sections = {data_name: set() for data_name in self.data_names}
		channels_by_section = {
			data_name: self._build_channels(data_name=data_name, rot_format=rot_format)
			for data_name in self.data_names
		}
		if not keys:
			for data_name, channels in channels_by_section.items():
				default_keys = {
					channel["key"]
					for channel in channels
					if channel["key"].endswith(".root_pos.x")
					or channel["key"].endswith(".root_pos.y")
					or channel["key"].endswith(".root_pos.z")
				}
				sections[data_name] = default_keys
			return sections

		requested = set(keys)
		for data_name, channels in channels_by_section.items():
			for channel in channels:
				if channel["key"] in requested or channel["label"] in requested:
					sections[data_name].add(channel["key"])
					continue
				if any(channel["key"].startswith(prefix) or channel["label"].startswith(prefix) for prefix in requested):
					sections[data_name].add(channel["key"])
		return sections

	def _build_channels(self, data_name: str, rot_format: str | None = None) -> list[dict[str, Any]]:
		section = self.data_section(data_name)
		self._normalize_section(section)
		length = int(section.get("length", 0))
		if length == 0:
			return []

		channels: list[dict[str, Any]] = []
		joint_names = list(self.basic.get("joint_names", []))
		joint_dims = list(self.basic.get("joint_dims", []))
		link_names = list(self.basic.get("link_names", []))
		imu_names = list(self.basic.get("imu_names", []))
		time_names = list(self.basic.get("time_names", []))
		fps = int(section.get("fps", 0) or 0)
		x_values = np.arange(length, dtype=np.float64) / float(fps) if fps > 0 else np.arange(length, dtype=np.float64)

		root_pos = _as_float_array(section.get("root_pos", []), width=3)
		if root_pos.shape[0] > 0:
			for axis_index, axis_name in enumerate(("x", "y", "z")):
				channels.append({
					"key": f"{data_name}.root_pos.{axis_name}",
					"label": f"{data_name}.root_pos.{axis_name}",
					"values": root_pos[:, axis_index],
					"x_values": x_values,
				})

		root_rot = _as_float_array(section.get("root_rot", []), width=4)
		if root_rot.shape[0] > 0:
			if rot_format in {None, "both", "rotvec"}:
				stable_quat = root_rot.copy()
				for index in range(1, stable_quat.shape[0]):
					if float(np.dot(stable_quat[index - 1], stable_quat[index])) < 0.0:
						stable_quat[index] = -stable_quat[index]
				rotvec = np.asarray([_quaternion_wxyz_to_rotvec(quat) for quat in stable_quat], dtype=np.float64)
				for axis_index, axis_name in enumerate(("x", "y", "z")):
					channels.append({
						"key": f"{data_name}.root_rot.rotvec.{axis_name}",
						"label": f"{data_name}.root_rot.rotvec.{axis_name}",
						"values": rotvec[:, axis_index],
						"x_values": x_values,
					})
			if rot_format in {None, "both", "euler"}:
				euler = np.asarray([_quaternion_wxyz_to_euler_xyz(quat) for quat in root_rot], dtype=np.float64)
				euler[:, 2] = np.unwrap(euler[:, 2]) if euler.shape[0] > 1 else euler[:, 2]
				for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
					channels.append({
						"key": f"{data_name}.root_rot.euler.{axis_name}",
						"label": f"{data_name}.root_rot.euler.{axis_name}",
						"values": euler[:, axis_index],
						"x_values": x_values,
					})

		joint_group = section.get("joint", {})
		joint_total_dim = int(sum(joint_dims))
		for field_name in ("pos", "vel", "acc", "tau"):
			values = _as_float_array(joint_group.get(field_name, []), width=joint_total_dim) if joint_total_dim > 0 else np.zeros((0, 0), dtype=np.float64)
			if values.shape[0] == 0:
				continue
			start = 0
			for joint_name, joint_dim in zip(joint_names, joint_dims):
				joint_slice = values[:, start : start + joint_dim]
				if joint_dim == 1:
					channels.append({
						"key": f"{data_name}.joint.{field_name}.{joint_name}",
						"label": f"{data_name}.joint.{field_name}.{joint_name}",
						"values": joint_slice[:, 0],
						"x_values": x_values,
					})
				else:
					for axis_index in range(joint_dim):
						channels.append({
							"key": f"{data_name}.joint.{field_name}.{joint_name}[{axis_index}]",
							"label": f"{data_name}.joint.{field_name}.{joint_name}[{axis_index}]",
							"values": joint_slice[:, axis_index],
							"x_values": x_values,
						})
				start += joint_dim

		link_group = section.get("link", {})
		link_specs = {
			"pos": (3, ("x", "y", "z")),
			"rot": (4, ("w", "x", "y", "z")),
			"lin_vel": (3, ("x", "y", "z")),
			"ang_vel": (3, ("x", "y", "z")),
		}
		for field_name, (width, axes) in link_specs.items():
			values = np.asarray(link_group.get(field_name, []), dtype=np.float64)
			if values.size == 0:
				continue
			values = values.reshape((values.shape[0], len(link_names), width))
			for link_index, link_name in enumerate(link_names):
				for axis_index, axis_name in enumerate(axes):
					channels.append({
						"key": f"{data_name}.link.{field_name}.{link_name}.{axis_name}",
						"label": f"{data_name}.link.{field_name}.{link_name}.{axis_name}",
						"values": values[:, link_index, axis_index],
						"x_values": x_values,
					})

		imu_group = section.get("imu", {})
		imu_specs = {
			"pos": (3, ("x", "y", "z")),
			"rot": (4, ("w", "x", "y", "z")),
			"gyro": (3, ("x", "y", "z")),
			"acc": (3, ("x", "y", "z")),
			"lin_vel": (3, ("x", "y", "z")),
		}
		for field_name, (width, axes) in imu_specs.items():
			values = np.asarray(imu_group.get(field_name, []), dtype=np.float64)
			if values.size == 0:
				continue
			values = values.reshape((values.shape[0], len(imu_names), width))
			for imu_index, imu_name in enumerate(imu_names):
				for axis_index, axis_name in enumerate(axes):
					channels.append({
						"key": f"{data_name}.imu.{field_name}.{imu_name}.{axis_name}",
						"label": f"{data_name}.imu.{field_name}.{imu_name}.{axis_name}",
						"values": values[:, imu_index, axis_index],
						"x_values": x_values,
					})

		time_values = _normalize_time_array(
			section.get("time", []),
			time_name_count=len(time_names),
		)
		if time_values.size > 0:
			inferred_names = time_names if time_values.shape[1] == len(time_names) else [f"time_{index}" for index in range(time_values.shape[1])]
			for time_index, time_name in enumerate(inferred_names):
				channels.append({
					"key": f"{data_name}.time.{time_name}",
					"label": f"{data_name}.time.{time_name}",
					"values": time_values[:, time_index],
					"x_values": x_values,
				})

		return channels


def load(path: str | Path) -> OMF:
	path = Path(path)
	suffix = path.suffix.lower()
	if suffix == ".msgpack":
		data = msgpack.unpackb(path.read_bytes(), raw=False)
	elif suffix in {".yaml", ".yml"}:
		data = yaml.safe_load(path.read_text(encoding="utf-8"))
	elif suffix == ".json":
		data = json.loads(path.read_text(encoding="utf-8"))
	else:
		raise ValueError(f"Unsupported input format: {suffix}")
	if not isinstance(data, dict):
		raise ValueError("OMF payload must be a dict")
	return OMF(data=data)
