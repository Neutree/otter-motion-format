from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import yaml
from datetime import datetime


def _deep_convert_for_storage(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		kv = {}
		for key, sub_value in value.items():
			if key == "date" and isinstance(sub_value, datetime):
				kv[key] = sub_value.strftime("%Y%m%dT%H:%M:%S.%f")
			else:
				kv[key] = _deep_convert_for_storage(sub_value)
		return kv
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


def _resample_linear_array(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=np.float64)
	if values.shape[0] == 0:
		return values.copy()
	if target_times.size == 0:
		return np.zeros((0, *values.shape[1:]), dtype=np.float64)
	if values.shape[0] == 1:
		return np.repeat(values[:1], target_times.size, axis=0)
	flat_values = values.reshape((values.shape[0], -1))
	resampled = np.empty((target_times.size, flat_values.shape[1]), dtype=np.float64)
	for column in range(flat_values.shape[1]):
		resampled[:, column] = np.interp(
			target_times,
			source_times,
			flat_values[:, column],
			left=flat_values[0, column],
			right=flat_values[-1, column],
		)
	return resampled.reshape((target_times.size, *values.shape[1:]))


def _resample_quaternion_array(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=np.float64)
	if values.shape[0] == 0:
		return values.copy()
	if target_times.size == 0:
		return np.zeros((0, *values.shape[1:]), dtype=np.float64)
	groups = values.reshape((values.shape[0], -1, 4)).copy()
	if groups.shape[0] == 1:
		result = np.repeat(groups[:1], target_times.size, axis=0)
	else:
		for group_index in range(groups.shape[1]):
			for frame_index in range(1, groups.shape[0]):
				if float(np.dot(groups[frame_index - 1, group_index], groups[frame_index, group_index])) < 0.0:
					groups[frame_index, group_index] = -groups[frame_index, group_index]
		result = np.empty((target_times.size, groups.shape[1], 4), dtype=np.float64)
		for group_index in range(groups.shape[1]):
			for component_index in range(4):
				result[:, group_index, component_index] = np.interp(
					target_times,
					source_times,
					groups[:, group_index, component_index],
					left=groups[0, group_index, component_index],
					right=groups[-1, group_index, component_index],
				)
	norms = np.linalg.norm(result, axis=2, keepdims=True)
	norms[norms <= 1e-12] = 1.0
	result = result / norms
	return result.reshape((target_times.size, *values.shape[1:]))


def _quaternion_xyzw_to_euler_xyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
	x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
	sinr_cosp = 2.0 * (w * x + y * z)
	cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2.0 * (w * y - z * x)
	pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)

	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)
	return np.array([roll, pitch, yaw], dtype=np.float64)


def _quaternion_xyzw_to_rotvec(quaternion_xyzw: np.ndarray) -> np.ndarray:
	quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
	norm = np.linalg.norm(quaternion_xyzw)
	if norm <= 1e-12:
		return np.zeros(3, dtype=np.float64)
	quaternion_xyzw = quaternion_xyzw / norm
	xyz = quaternion_xyzw[:3]
	w = float(np.clip(quaternion_xyzw[3], -1.0, 1.0))
	angle = 2.0 * np.arccos(w)
	sin_half = np.sqrt(max(0.0, 1.0 - w * w))
	if sin_half <= 1e-12:
		return np.zeros(3, dtype=np.float64)
	axis = xyz / sin_half
	return axis * angle


def _stabilize_quaternion_xyzw_batch(quaternions: np.ndarray) -> np.ndarray:
	stable = np.asarray(quaternions, dtype=np.float64).copy()
	if stable.shape[0] <= 1:
		return stable
	dots = np.sum(stable[1:] * stable[:-1], axis=1)
	signs = np.cumprod(np.where(dots < 0.0, -1.0, 1.0))
	stable[1:] *= signs[:, None]
	return stable


def _quaternion_xyzw_to_euler_xyz_batch(quaternions: np.ndarray) -> np.ndarray:
	quaternions = np.asarray(quaternions, dtype=np.float64)
	if quaternions.size == 0:
		return np.zeros((0, 3), dtype=np.float64)
	x = quaternions[:, 0]
	y = quaternions[:, 1]
	z = quaternions[:, 2]
	w = quaternions[:, 3]
	sinr_cosp = 2.0 * (w * x + y * z)
	cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)
	sinp = 2.0 * (w * y - z * x)
	pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(np.clip(sinp, -1.0, 1.0)))
	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)
	euler = np.column_stack((roll, pitch, yaw))
	if euler.shape[0] > 1:
		euler[:, 2] = np.unwrap(euler[:, 2])
	return euler


def _quaternion_xyzw_to_rotvec_batch(quaternions: np.ndarray) -> np.ndarray:
	quaternions = np.asarray(quaternions, dtype=np.float64)
	if quaternions.size == 0:
		return np.zeros((0, 3), dtype=np.float64)
	norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
	safe = norms > 1e-12
	normalized = np.divide(quaternions, norms, out=np.zeros_like(quaternions), where=safe)
	xyz = normalized[:, :3]
	w = np.clip(normalized[:, 3], -1.0, 1.0)
	angle = 2.0 * np.arccos(w)
	sin_half = np.sqrt(np.maximum(0.0, 1.0 - w * w))
	axis = np.divide(xyz, sin_half[:, None], out=np.zeros_like(xyz), where=sin_half[:, None] > 1e-12)
	rotvec = axis * angle[:, None]
	rotvec[~safe[:, 0]] = 0.0
	return rotvec


def _copy_frame_value(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, list):
		if not value:
			return []
		first = value[0]
		if isinstance(first, list):
			return [list(item) if isinstance(item, list) else item for item in value]
		if isinstance(first, np.ndarray):
			return [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
		return list(value)
	return _deep_copy_data(value)

def msgpack_encode_datetime(obj):
    if isinstance(obj, datetime):
        return obj.strftime("%Y%m%dT%H:%M:%S.%f")
    return obj


# def msgpack_decode_datetime(obj):
#     if '__datetime__' in obj:
#         obj = datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
#     return obj

class OMF:
	VERSION = 1
	FORMAT = "omf"
	DEFAULT_DATA_NAMES = ["target", "actual"]
	_RESERVED_TOP_LEVEL_KEYS = {"version", "format", "basic"}
	# Recommended starter fields; users may add any extra keys under joint/link/imu.
	DEFAULT_JOINT_FIELDS = ("pos", "vel", "acc", "tau", "temp", "voltage", "current")
	DEFAULT_LINK_FIELD_SPECS: dict[str, tuple[int, tuple[str, ...]]] = {
		"pos": (3, ("x", "y", "z")),
		"rot": (4, ("x", "y", "z", "w")),
		"lin_vel": (3, ("x", "y", "z")),
		"ang_vel": (3, ("x", "y", "z")),
	}
	DEFAULT_IMU_FIELD_SPECS: dict[str, tuple[int, tuple[str, ...]]] = {
		"pos": (3, ("x", "y", "z")),
		"rot": (4, ("x", "y", "z", "w")),
		"gyro": (3, ("x", "y", "z")),
		"acc": (3, ("x", "y", "z")),
		"lin_vel": (3, ("x", "y", "z")),
	}
	_QUATERNION_FIELD_NAMES = frozenset({"rot"})

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
		date: str | datetime | None = None,
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
					"date": date if isinstance(date, (str, datetime)) else datetime.now(),
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
		self._ensure_date_as_datetime()
		self.validate()

	def _ensure_date_as_datetime(self) -> None:
		basic = self.data["basic"]
		date_value = basic.get("date")
		if isinstance(date_value, str):
			try:
				basic["date"] = datetime.strptime(date_value, "%Y%m%dT%H:%M:%S.%f")
			except ValueError:
				print(f"Warning: failed to parse date string: {date_value!r}, setting date to None")
				basic["date"] = None
		elif not isinstance(date_value, datetime):
			basic["date"] = None

	@classmethod
	def _empty_section(cls) -> dict[str, Any]:
		return {
			"fps": 0,
			"length": 0,
			"root_pos": [],
			"root_rot": [],
			"joint": {field_name: [] for field_name in cls.DEFAULT_JOINT_FIELDS},
			"link": {field_name: [] for field_name in cls.DEFAULT_LINK_FIELD_SPECS},
			"imu": {field_name: [] for field_name in cls.DEFAULT_IMU_FIELD_SPECS},
			"time": [],
		}

	@staticmethod
	def _axis_names_for_width(width: int) -> tuple[str, ...]:
		if width == 3:
			return ("x", "y", "z")
		if width == 4:
			return ("x", "y", "z", "w")
		return tuple(str(index) for index in range(width))

	@classmethod
	def _resolve_entity_field_spec(
		cls,
		field_name: str,
		values: Any,
		*,
		length: int,
		entity_count: int,
		known_specs: dict[str, tuple[int, tuple[str, ...]]],
	) -> tuple[int, tuple[str, ...]] | None:
		if field_name in known_specs:
			return known_specs[field_name]
		if entity_count <= 0 or length <= 0:
			return None
		raw = np.asarray(values, dtype=np.float64)
		if raw.size == 0:
			return None
		if raw.ndim == 3 and raw.shape[0] == length and raw.shape[1] == entity_count:
			width = int(raw.shape[2])
			return width, cls._axis_names_for_width(width)
		if raw.ndim == 2 and raw.shape[0] == length and raw.shape[1] % entity_count == 0:
			width = int(raw.shape[1] // entity_count)
			return width, cls._axis_names_for_width(width)
		if raw.ndim == 1 and length == 1 and raw.shape[0] % entity_count == 0:
			width = int(raw.shape[0] // entity_count)
			return width, cls._axis_names_for_width(width)
		return None

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
				if key in ("joint", "link", "imu"):
					group = section.setdefault(key, {})
					if not isinstance(group, dict):
						section[key] = _deep_copy_data(value)
						continue
					for field_name, field_default in value.items():
						group.setdefault(field_name, _deep_copy_data(field_default))
				else:
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
		for group_name in ("joint", "link", "imu"):
			group = section.get(group_name, {})
			if not isinstance(group, dict):
				raise ValueError(f"{data_name}.{group_name} must be a dict of named fields")
			for key, value in group.items():
				if not isinstance(value, (list, np.ndarray)):
					raise ValueError(f"{data_name}.{group_name}.{key} must be a list or array")
				if len(value) not in (0, length):
					raise ValueError(
						f"{data_name}.{group_name}.{key} length mismatch: expected 0 or {length}, got {len(value)}"
					)

	def summary(self) -> str:
		self.validate()
		lines = [
			f"OMF(name={self.name!r}, robot={self.basic.get('robot', '')!r})",
			f"  version: {self.data.get('version')}",
			f"  date: {self.basic.get('date')}",
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

	def append_frame(self, frame_data: dict[str, dict[str, Any]], *, validate: bool = True) -> None:
		for data_name, payload in frame_data.items():
			section = self.data_section(data_name)
			self._append_section_frame(section, payload)
		if validate:
			self.validate()

	def truncate_frames(self, max_frames: int, *, data_names: list[str] | None = None, validate: bool = True) -> None:
		if max_frames <= 0:
			return
		for data_name in (data_names or self.data_names):
			section = self.data_section(data_name)
			self._truncate_section(section, max_frames)
		if validate:
			self.validate()

	def chart_topology_signature(self, rot_format: str | None = None) -> tuple[Any, ...]:
		signature: list[Any] = [rot_format]
		for data_name in self.data_names:
			section = self.data_section(data_name)
			signature.append(
				(
					data_name,
					int(section.get("fps", 0) or 0),
					bool(section.get("root_pos")),
					bool(section.get("root_rot")),
					tuple(field_name for field_name, value in section.get("joint", {}).items() if value),
					tuple(field_name for field_name, value in section.get("link", {}).items() if value),
					tuple(field_name for field_name, value in section.get("imu", {}).items() if value),
					bool(section.get("time")),
					tuple(self.basic.get("joint_names", [])),
					tuple(int(dim) for dim in self.basic.get("joint_dims", [])),
					tuple(self.basic.get("link_names", [])),
					tuple(self.basic.get("imu_names", [])),
					tuple(self.basic.get("time_names", [])),
				)
			)
		return tuple(signature)

	def _append_section_frame(self, section: dict[str, Any], payload: dict[str, Any]) -> None:
		if "root_pos" in payload:
			section["root_pos"].append(_copy_frame_value(payload["root_pos"]))
		if "root_rot" in payload:
			section["root_rot"].append(_copy_frame_value(payload["root_rot"]))
		if "time" in payload:
			section["time"].append(_copy_frame_value(payload["time"]))

		for group_name in ("joint", "link", "imu"):
			group_payload = payload.get(group_name)
			if not isinstance(group_payload, dict):
				continue
			for field_name, field_value in group_payload.items():
				if field_name not in section[group_name]:
					section[group_name][field_name] = []
				section[group_name][field_name].append(_copy_frame_value(field_value))

		section["length"] = max(
			int(section.get("length", 0) or 0) + 1,
			len(section.get("root_pos", []) or []),
			len(section.get("root_rot", []) or []),
			len(section.get("time", []) or []),
			*(
				len(value)
				for group_name in ("joint", "link", "imu")
				for value in section.get(group_name, {}).values()
				if isinstance(value, list)
			),
		)

	def _truncate_section(self, section: dict[str, Any], max_frames: int) -> None:
		for key in ("root_pos", "root_rot", "time"):
			value = section.get(key, [])
			if isinstance(value, list) and len(value) > max_frames:
				section[key] = value[-max_frames:]
		for group_name in ("joint", "link", "imu"):
			group = section.get(group_name, {})
			for field_name, value in group.items():
				if isinstance(value, list) and len(value) > max_frames:
					group[field_name] = value[-max_frames:]
		section["length"] = self._section_length(section)

	def save(self, path: str | Path) -> None:
		self.validate()
		path = Path(path)
		payload = _deep_convert_for_storage(self.data)
		suffix = path.suffix.lower()
		if suffix == ".msgpack":
			packed = msgpack.packb(payload, default=msgpack_encode_datetime, use_bin_type=True)
			path.write_bytes(packed)
			return
		if suffix in {".yaml", ".yml"}:
			path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
			return
		if suffix == ".json":
			path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
			return
		raise ValueError(f"Unsupported output format: {suffix}")

	def build_chart_payload(
		self,
		rot_format: str | None = None,
		*,
		channel_keys: set[str] | None = None,
		include_values: bool = True,
		validate: bool = True,
	) -> dict[str, list[dict[str, Any]]]:
		if validate:
			self.validate()
		return {
			data_name: self._build_channels(
				data_name=data_name,
				rot_format=rot_format,
				include_keys=channel_keys,
				include_values=include_values,
			)
			for data_name in self.data_names
		}

	def resolve_chart_preselected(self, keys: list[str] | None = None, rot_format: str | None = None) -> dict[str, set[str]]:
		return self._resolve_preselected_keys(keys=keys, rot_format=rot_format)

	def show_chart(self, keys: list[str] | None = None, rot_format: str | None = None) -> None:
		self.validate()
		try:
			from .viewer import build_channel_specs, show_omf_viewer
		except ImportError as exc:
			raise ImportError(
				"Visualization requires optional dependencies. Install with: pip install 'otter-motion-format[viz]'"
			) from exc

		sections = build_channel_specs(self.build_chart_payload(rot_format=rot_format))
		preselected = self.resolve_chart_preselected(keys=keys, rot_format=rot_format)
		show_omf_viewer(
			title=self.name or "OMF Viewer",
			sections=sections,
			preselected=preselected,
			layer_styles=self._default_layer_styles(),
			raw_data=self.to_dict(),
		)

	def resampled(self, target_fps: int | str = "max") -> "OMF":
		self.validate()
		if target_fps == "max":
			resolved_fps = max((int(self.data_section(data_name).get("fps", 0) or 0) for data_name in self.data_names), default=0)
		else:
			resolved_fps = int(target_fps)
		if resolved_fps <= 0:
			return self.clone()

		result = self.clone()
		joint_total_dim = int(sum(result.basic.get("joint_dims", [])))
		link_count = len(result.basic.get("link_names", []))
		imu_count = len(result.basic.get("imu_names", []))
		time_name_count = len(result.basic.get("time_names", []))

		for data_name in result.data_names:
			section = result.data_section(data_name)
			length = int(section.get("length", 0) or 0)
			source_fps = int(section.get("fps", 0) or 0)
			if length <= 0 or source_fps <= 0 or source_fps == resolved_fps:
				continue
			duration = float(length - 1) / float(max(source_fps, 1))
			target_length = max(int(round(duration * resolved_fps)) + 1, 1)
			source_times = np.arange(length, dtype=np.float64) / float(max(source_fps, 1))
			target_times = np.arange(target_length, dtype=np.float64) / float(resolved_fps)

			root_pos = _as_float_array(section.get("root_pos", []), width=3)
			section["root_pos"] = _resample_linear_array(root_pos, source_times, target_times).tolist() if root_pos.shape[0] > 0 else []

			root_rot = _as_float_array(section.get("root_rot", []), width=4)
			section["root_rot"] = _resample_quaternion_array(root_rot, source_times, target_times).tolist() if root_rot.shape[0] > 0 else []

			joint_group = section.setdefault("joint", {})
			for field_name, field_values in list(joint_group.items()):
				if not field_values:
					joint_group[field_name] = []
					continue
				joint_values = (
					_as_float_array(field_values, width=joint_total_dim)
					if joint_total_dim > 0
					else np.zeros((0, 0), dtype=np.float64)
				)
				joint_group[field_name] = (
					_resample_linear_array(joint_values, source_times, target_times).tolist()
					if joint_values.shape[0] > 0
					else []
				)

			link_group = section.setdefault("link", {})
			for field_name, field_values in list(link_group.items()):
				values = np.asarray(field_values, dtype=np.float64)
				if values.size == 0 or link_count == 0:
					link_group[field_name] = []
					continue
				spec = self._resolve_entity_field_spec(
					field_name,
					values,
					length=length,
					entity_count=link_count,
					known_specs=self.DEFAULT_LINK_FIELD_SPECS,
				)
				if spec is None:
					raise ValueError(
						f"Cannot infer shape for {data_name}.link.{field_name} during resample "
						f"(expected length={length}, links={link_count})"
					)
				width, _ = spec
				values = values.reshape((length, link_count, width))
				resampled_values = (
					_resample_quaternion_array(values, source_times, target_times)
					if field_name in self._QUATERNION_FIELD_NAMES
					else _resample_linear_array(values, source_times, target_times)
				)
				link_group[field_name] = resampled_values.tolist()

			imu_group = section.setdefault("imu", {})
			for field_name, field_values in list(imu_group.items()):
				values = np.asarray(field_values, dtype=np.float64)
				if values.size == 0 or imu_count == 0:
					imu_group[field_name] = []
					continue
				spec = self._resolve_entity_field_spec(
					field_name,
					values,
					length=length,
					entity_count=imu_count,
					known_specs=self.DEFAULT_IMU_FIELD_SPECS,
				)
				if spec is None:
					raise ValueError(
						f"Cannot infer shape for {data_name}.imu.{field_name} during resample "
						f"(expected length={length}, imus={imu_count})"
					)
				width, _ = spec
				values = values.reshape((length, imu_count, width))
				resampled_values = (
					_resample_quaternion_array(values, source_times, target_times)
					if field_name in self._QUATERNION_FIELD_NAMES
					else _resample_linear_array(values, source_times, target_times)
				)
				imu_group[field_name] = resampled_values.tolist()

			time_values = _normalize_time_array(section.get("time", []), time_name_count=time_name_count)
			section["time"] = _resample_linear_array(time_values, source_times, target_times).tolist() if time_values.shape[0] > 0 else []
			section["fps"] = resolved_fps
			section["length"] = target_length

		result.validate()
		return result

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
			data_name: self._build_channels(data_name=data_name, rot_format=rot_format, include_values=False)
			for data_name in self.data_names
		}
		if not keys:
			for data_name, channels in channels_by_section.items():
				default_keys = {
					channel["key"]
					for channel in channels
					if str(channel["key"]).endswith(".root_rot.euler.pitch")
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

	def _build_channels(
		self,
		data_name: str,
		rot_format: str | None = None,
		*,
		include_keys: set[str] | None = None,
		include_values: bool = True,
	) -> list[dict[str, Any]]:
		section = self.data_section(data_name)
		length = int(section.get("length", 0) or 0)
		if length <= 0 and not any(
			bool(section.get(key))
			for key in ("root_pos", "root_rot", "time")
		) and not any(bool(value) for group_name in ("joint", "link", "imu") for value in section.get(group_name, {}).values()):
			# Prefer length, but catalog can also appear from non-empty arrays before length is synced.
			computed = self._section_length(section)
			length = computed
			section["length"] = computed
		if length == 0:
			return []

		channels: list[dict[str, Any]] = []
		joint_names = list(self.basic.get("joint_names", []))
		joint_dims = list(self.basic.get("joint_dims", []))
		link_names = list(self.basic.get("link_names", []))
		imu_names = list(self.basic.get("imu_names", []))
		time_names = list(self.basic.get("time_names", []))
		fps = int(section.get("fps", 0) or 0)
		empty_x = np.zeros(0, dtype=np.float64)
		empty_y = np.zeros(0, dtype=np.float64)
		x_values = (
			(np.arange(length, dtype=np.float64) / float(fps) if fps > 0 else np.arange(length, dtype=np.float64))
			if include_values
			else empty_x
		)

		def want(key: str) -> bool:
			return include_keys is None or key in include_keys

		def add_channel(key: str, label: str, values: np.ndarray | None = None) -> None:
			if not want(key):
				return
			channels.append(
				{
					"key": key,
					"label": label,
					"values": empty_y if (not include_values or values is None) else values,
					"x_values": x_values if include_values else empty_x,
				}
			)

		root_pos_present = bool(section.get("root_pos"))
		if root_pos_present:
			root_pos = _as_float_array(section.get("root_pos", []), width=3) if include_values else None
			for axis_index, axis_name in enumerate(("x", "y", "z")):
				add_channel(
					f"{data_name}.root_pos.{axis_name}",
					f"{data_name}.root_pos.{axis_name}",
					None if root_pos is None else root_pos[:, axis_index],
				)

		root_rot_present = bool(section.get("root_rot"))
		if root_rot_present:
			need_rotvec = rot_format in {None, "both", "rotvec"} and (
				include_keys is None or any(key.startswith(f"{data_name}.root_rot.rotvec.") for key in include_keys)
			)
			need_euler = rot_format in {None, "both", "euler"} and (
				include_keys is None or any(key.startswith(f"{data_name}.root_rot.euler.") for key in include_keys)
			)
			root_rot = None
			if include_values and (need_rotvec or need_euler):
				root_rot = _as_float_array(section.get("root_rot", []), width=4)
			if rot_format in {None, "both", "rotvec"}:
				rotvec = None
				if include_values and need_rotvec and root_rot is not None and root_rot.shape[0] > 0:
					rotvec = _quaternion_xyzw_to_rotvec_batch(_stabilize_quaternion_xyzw_batch(root_rot))
				for axis_index, axis_name in enumerate(("x", "y", "z")):
					add_channel(
						f"{data_name}.root_rot.rotvec.{axis_name}",
						f"{data_name}.root_rot.rotvec.{axis_name}",
						None if rotvec is None else rotvec[:, axis_index],
					)
			if rot_format in {None, "both", "euler"}:
				euler = None
				if include_values and need_euler and root_rot is not None and root_rot.shape[0] > 0:
					euler = _quaternion_xyzw_to_euler_xyz_batch(root_rot)
				for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
					add_channel(
						f"{data_name}.root_rot.euler.{axis_name}",
						f"{data_name}.root_rot.euler.{axis_name}",
						None if euler is None else euler[:, axis_index],
					)

		joint_group = section.get("joint", {})
		joint_total_dim = int(sum(joint_dims))
		for field_name, field_values in joint_group.items():
			if not field_values:
				continue
			field_prefix = f"{data_name}.joint.{field_name}."
			need_field = include_keys is None or any(key.startswith(field_prefix) for key in include_keys)
			values = None
			if include_values and need_field and joint_total_dim > 0:
				values = _as_float_array(field_values, width=joint_total_dim)
				if values.shape[0] == 0:
					continue
			elif include_values and not need_field:
				continue
			start = 0
			for joint_name, joint_dim in zip(joint_names, joint_dims):
				if joint_dim == 1:
					key = f"{data_name}.joint.{field_name}.{joint_name}"
					add_channel(
						key,
						key,
						None if values is None else values[:, start],
					)
				else:
					for axis_index in range(joint_dim):
						key = f"{data_name}.joint.{field_name}.{joint_name}[{axis_index}]"
						add_channel(
							key,
							key,
							None if values is None else values[:, start + axis_index],
						)
				start += joint_dim

		link_group = section.get("link", {})
		for field_name, field_values in link_group.items():
			if not field_values or not link_names:
				continue
			spec = self._resolve_entity_field_spec(
				field_name,
				field_values,
				length=length,
				entity_count=len(link_names),
				known_specs=self.DEFAULT_LINK_FIELD_SPECS,
			)
			if spec is None:
				continue
			width, axes = spec
			field_prefix = f"{data_name}.link.{field_name}."
			need_field = include_keys is None or any(key.startswith(field_prefix) for key in include_keys)
			values = None
			if include_values and need_field:
				raw = np.asarray(field_values, dtype=np.float64)
				if raw.size == 0:
					continue
				values = raw.reshape((raw.shape[0], len(link_names), width))
			elif include_values and not need_field:
				continue
			for link_index, link_name in enumerate(link_names):
				for axis_index, axis_name in enumerate(axes):
					key = f"{data_name}.link.{field_name}.{link_name}.{axis_name}"
					add_channel(
						key,
						key,
						None if values is None else values[:, link_index, axis_index],
					)

		imu_group = section.get("imu", {})
		for field_name, field_values in imu_group.items():
			if not field_values or not imu_names:
				continue
			spec = self._resolve_entity_field_spec(
				field_name,
				field_values,
				length=length,
				entity_count=len(imu_names),
				known_specs=self.DEFAULT_IMU_FIELD_SPECS,
			)
			if spec is None:
				continue
			width, axes = spec
			field_prefix = f"{data_name}.imu.{field_name}."
			need_field = include_keys is None or any(key.startswith(field_prefix) for key in include_keys)
			values = None
			if include_values and need_field:
				raw = np.asarray(field_values, dtype=np.float64)
				if raw.size == 0:
					continue
				values = raw.reshape((raw.shape[0], len(imu_names), width))
			elif include_values and not need_field:
				continue
			for imu_index, imu_name in enumerate(imu_names):
				for axis_index, axis_name in enumerate(axes):
					key = f"{data_name}.imu.{field_name}.{imu_name}.{axis_name}"
					add_channel(
						key,
						key,
						None if values is None else values[:, imu_index, axis_index],
					)

		if section.get("time"):
			time_values = None
			if include_values and (include_keys is None or any(key.startswith(f"{data_name}.time.") for key in include_keys)):
				time_values = _normalize_time_array(
					section.get("time", []),
					time_name_count=len(time_names),
				)
			inferred_names = time_names
			if time_values is not None and time_values.size > 0:
				inferred_names = (
					time_names
					if time_values.shape[1] == len(time_names)
					else [f"time_{index}" for index in range(time_values.shape[1])]
				)
			elif not inferred_names:
				# Catalog path: infer column count without full validate/normalize.
				raw_time = np.asarray(section.get("time", []), dtype=np.float64)
				if raw_time.ndim == 2 and raw_time.shape[1] > 0:
					inferred_names = [f"time_{index}" for index in range(raw_time.shape[1])]
				else:
					inferred_names = time_names or ["time_0"]
			for time_index, time_name in enumerate(inferred_names):
				key = f"{data_name}.time.{time_name}"
				add_channel(
					key,
					key,
					None if time_values is None or time_values.size == 0 else time_values[:, time_index],
				)

		return channels


def load(path: str | Path, *, resample_fps: int | str | None = None) -> OMF:
	path = Path(path)
	suffix = path.suffix.lower()
	if suffix == ".msgpack":
		# data = msgpack.unpackb(path.read_bytes(), object_hook=msgpack_decode_datetime, raw=False)
		data = msgpack.unpackb(path.read_bytes(), raw=False)
	elif suffix in {".yaml", ".yml"}:
		data = yaml.safe_load(path.read_text(encoding="utf-8"))
	elif suffix == ".json":
		data = json.loads(path.read_text(encoding="utf-8"))
	else:
		raise ValueError(f"Unsupported input format: {suffix}")
	if not isinstance(data, dict):
		raise ValueError("OMF payload must be a dict")
	motion = OMF(data=data)
	if resample_fps is not None:
		return motion.resampled(resample_fps)
	return motion
