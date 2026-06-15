from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from otter_motion_format.otter_motion_format import OMF, load


def _convert_quaternion_array(values: Any, *, width_prefix: tuple[int, ...]) -> tuple[Any, int]:
	array = np.asarray(values, dtype=np.float64)
	if array.size == 0:
		return values, 0
	shape = array.shape
	if len(shape) != len(width_prefix) + 1 or shape[:-1] != width_prefix or shape[-1] != 4:
		frame_count = shape[0] if len(shape) > 0 else 0
		reshaped = array.reshape((frame_count, *width_prefix, 4))
	else:
		reshaped = array
	converted = np.concatenate([reshaped[..., 1:], reshaped[..., :1]], axis=-1)
	return converted.tolist(), int(converted.shape[0])


def _convert_section_quaternions(payload: dict[str, Any], data_name: str, *, link_count: int, imu_count: int) -> dict[str, int]:
	section = payload[data_name]
	counts = {"root_rot": 0, "link.rot": 0, "imu.rot": 0}

	root_rot = section.get("root_rot", [])
	converted_root, root_frames = _convert_quaternion_array(root_rot, width_prefix=())
	section["root_rot"] = converted_root
	counts["root_rot"] = root_frames

	link_group = section.setdefault("link", {})
	link_rot = link_group.get("rot", [])
	converted_link, link_frames = _convert_quaternion_array(link_rot, width_prefix=(link_count,))
	link_group["rot"] = converted_link
	counts["link.rot"] = link_frames

	imu_group = section.setdefault("imu", {})
	imu_rot = imu_group.get("rot", [])
	converted_imu, imu_frames = _convert_quaternion_array(imu_rot, width_prefix=(imu_count,))
	imu_group["rot"] = converted_imu
	counts["imu.rot"] = imu_frames

	return counts


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="otter-omf-wxyz-to-xyzw",
		description="Convert legacy OMF quaternion fields from wxyz order to xyzw order.",
	)
	parser.add_argument("motion_file", type=str, help="Path to the source OMF file (.msgpack/.yaml/.yml/.json)")
	parser.add_argument(
		"-o",
		"--output",
		type=str,
		default=None,
		help="Optional output path. If omitted, the source file is overwritten.",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	source_path = Path(args.motion_file).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve() if args.output else source_path

	motion = load(source_path)
	payload = motion.to_dict()
	converted = OMF(data=payload)
	link_count = len(converted.basic.get("link_names", []))
	imu_count = len(converted.basic.get("imu_names", []))

	data_names = list(converted.data_names)
	stats: dict[str, dict[str, int]] = {}
	for data_name in data_names:
		stats[data_name] = _convert_section_quaternions(payload, data_name, link_count=link_count, imu_count=imu_count)

	OMF(data=payload).save(output_path)

	print(f"Converted legacy wxyz quaternions to xyzw: {source_path} -> {output_path}")
	for data_name in data_names:
		counts = stats[data_name]
		print(
			f"  {data_name}: root_rot={counts['root_rot']} frames, "
			f"link.rot={counts['link.rot']} frames, imu.rot={counts['imu.rot']} frames"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())