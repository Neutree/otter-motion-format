from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .otter_motion_format import OMF, _as_float_array, load


def _load_gmr_payload(path: Path) -> dict[str, Any]:
	with path.open("rb") as handle:
		payload = pickle.load(handle)
	if not isinstance(payload, dict):
		raise ValueError("GMR payload must be a dict")
	return payload


def _default_joint_names(count: int) -> list[str]:
	return [f"joint_{index}" for index in range(count)]


def _resolve_omf_data_name(motion: OMF, requested: str | None) -> str:
	if requested:
		if requested not in motion.data_names:
			raise ValueError(f"Unknown OMF data section: {requested}")
		return requested
	if "actual" in motion.data_names:
		return "actual"
	if motion.data_names:
		return motion.data_names[0]
	raise ValueError("OMF contains no data sections")


def convert_gmr_to_omf(
	gmr_path: str | Path,
	omf_path: str | Path,
	*,
	name: str | None = None,
	robot: str,
	data_name: str = "target",
) -> Path:
	gmr_path = Path(gmr_path).expanduser().resolve()
	omf_path = Path(omf_path).expanduser().resolve()
	payload = _load_gmr_payload(gmr_path)
	root_pos = _as_float_array(payload.get("root_pos", []), width=3)
	root_rot_xyzw = _as_float_array(payload.get("root_rot", []), width=4)
	dof_pos = np.asarray(payload.get("dof_pos", []), dtype=np.float64)
	if dof_pos.ndim == 1:
		dof_pos = dof_pos.reshape((-1, 1))
	frame_count = max(root_pos.shape[0], root_rot_xyzw.shape[0], dof_pos.shape[0])
	joint_names = list(payload.get("joint_names") or _default_joint_names(dof_pos.shape[1] if dof_pos.ndim == 2 else 0))
	if dof_pos.ndim == 2 and joint_names and dof_pos.shape[1] != len(joint_names):
		raise ValueError("joint_names length does not match dof_pos width")
	motion = OMF(
		name=name or gmr_path.stem,
		robot=robot,
		joint_names=joint_names,
		joint_dims=[1] * len(joint_names),
		data_names=[data_name],
	)
	section = motion.data_section(data_name)
	section["fps"] = int(float(payload.get("fps", 0) or 0))
	section["length"] = int(frame_count)
	section["root_pos"] = root_pos.tolist()
	section["root_rot"] = root_rot_xyzw[:, [3, 0, 1, 2]].tolist() if root_rot_xyzw.size > 0 else []
	section["joint"]["pos"] = dof_pos.tolist() if dof_pos.size > 0 else []
	motion.save(omf_path)
	return omf_path


def convert_omf_to_gmr(
	omf_path: str | Path,
	gmr_path: str | Path,
	*,
	data_name: str | None = None,
) -> Path:
	omf_path = Path(omf_path).expanduser().resolve()
	gmr_path = Path(gmr_path).expanduser().resolve()
	motion = load(omf_path)
	resolved_data_name = _resolve_omf_data_name(motion, data_name)
	section = motion.data_section(resolved_data_name)
	joint_names = list(motion.basic.get("joint_names", []))
	joint_dims = list(motion.basic.get("joint_dims", []))
	if any(int(dim) != 1 for dim in joint_dims):
		raise ValueError("OMF to GMR export currently requires joint_dims to be all 1")
	root_pos = _as_float_array(section.get("root_pos", []), width=3)
	root_rot_wxyz = _as_float_array(section.get("root_rot", []), width=4)
	joint_pos = np.asarray(section.get("joint", {}).get("pos", []), dtype=np.float64)
	if joint_pos.ndim == 1:
		joint_pos = joint_pos.reshape((-1, 1))
	if joint_pos.size == 0:
		joint_pos = np.zeros((int(section.get("length", 0)), len(joint_names)), dtype=np.float64)
	if joint_names and joint_pos.shape[1] != len(joint_names):
		raise ValueError("OMF joint.pos width does not match basic.joint_names length")
	payload = {
		"fps": int(section.get("fps", 0) or 0),
		"root_pos": root_pos,
		"root_rot": root_rot_wxyz[:, [1, 2, 3, 0]] if root_rot_wxyz.size > 0 else np.zeros((0, 4), dtype=np.float64),
		"dof_pos": joint_pos,
		"joint_names": joint_names,
	}
	with gmr_path.open("wb") as handle:
		pickle.dump(payload, handle)
	return gmr_path


def gmr_to_omf_main() -> int:
	parser = argparse.ArgumentParser(description="Convert a GMR pickle motion file into OMF")
	parser.add_argument("gmr_file", type=str, help="Input GMR pickle file")
	parser.add_argument("omf_file", type=str, help="Output OMF file (.msgpack/.yaml/.json)")
	parser.add_argument("--robot", required=True, help="Robot name written into the OMF basic section")
	parser.add_argument("--name", default=None, help="Motion name written into the OMF basic section")
	parser.add_argument("--data-name", default="target", help="OMF data section name to create")
	args = parser.parse_args()
	convert_gmr_to_omf(args.gmr_file, args.omf_file, name=args.name, robot=args.robot, data_name=args.data_name)
	return 0


def omf_to_gmr_main() -> int:
	parser = argparse.ArgumentParser(description="Convert an OMF motion file into a GMR pickle")
	parser.add_argument("omf_file", type=str, help="Input OMF file")
	parser.add_argument("gmr_file", type=str, help="Output GMR pickle file")
	parser.add_argument("--data-name", default=None, help="OMF data section to export, defaulting to actual then the first available section")
	args = parser.parse_args()
	convert_omf_to_gmr(args.omf_file, args.gmr_file, data_name=args.data_name)
	return 0