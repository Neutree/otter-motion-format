

from __future__ import annotations

import argparse
from pathlib import Path

from .otter_motion_format import load


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="otter-motion-format",
		description="Load an OMF motion file, print a summary, and optionally open the interactive viewer.",
	)
	parser.add_argument("motion_file", type=str, help="Path to the motion file (.msgpack/.yaml/.yml/.json)")
	parser.add_argument(
		"--keys",
		nargs="*",
		default=None,
		help="Optional channel prefixes or full channel names to preselect in the chart viewer",
	)
	parser.add_argument(
		"--rot-format",
		choices=["rotvec", "euler", "both"],
		default="both",
		help="Rotation display format used by the viewer",
	)
	parser.add_argument(
		"--summary-only",
		action="store_true",
		help="Print the summary and exit without opening the viewer",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	motion_path = Path(args.motion_file).expanduser().resolve()
	motion = load(motion_path)
	motion.summary()

	if not args.summary_only:
		motion.show_chart(keys=args.keys, rot_format=None if args.rot_format == "both" else args.rot_format)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

