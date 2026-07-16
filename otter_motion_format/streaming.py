from __future__ import annotations

import argparse
import queue
import signal
import socket
import struct
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any

import msgpack

from .otter_motion_format import OMF


_HEADER_STRUCT = struct.Struct("!I")


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
	chunks: list[bytes] = []
	remaining = size
	while remaining > 0:
		chunk = sock.recv(remaining)
		if not chunk:
			return None
		chunks.append(chunk)
		remaining -= len(chunk)
	return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any] | None:
	header = _recv_exact(sock, _HEADER_STRUCT.size)
	if header is None:
		return None
	(message_size,) = _HEADER_STRUCT.unpack(header)
	payload = _recv_exact(sock, message_size)
	if payload is None:
		return None
	return msgpack.unpackb(payload, raw=False)


def _send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
	packed = msgpack.packb(payload, use_bin_type=True)
	sock.sendall(_HEADER_STRUCT.pack(len(packed)))
	sock.sendall(packed)


class OMFStreamClient:
	def __init__(
		self,
		host: str = "127.0.0.1",
		port: int = 8765,
		send_interval_s: float = 0.2,
		max_batch_size: int = 16,
		queue_size: int = 256,
		connect_timeout_s: float = 1.0,
		log_prefix: str = "[otter-stream-client]",
	) -> None:
		self.host = str(host)
		self.port = int(port)
		self.send_interval_s = max(0.01, float(send_interval_s))
		self.max_batch_size = max(1, int(max_batch_size))
		self.connect_timeout_s = max(0.1, float(connect_timeout_s))
		self.log_prefix = str(log_prefix)
		self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, int(queue_size)))
		self._init_payload: dict[str, Any] | None = None
		self._thread: threading.Thread | None = None
		self._stop_event = threading.Event()
		self._socket: socket.socket | None = None
		self._last_connect_error_time = 0.0

	def configure(self, payload: dict[str, Any]) -> None:
		self._init_payload = dict(payload)
		self._ensure_thread_started()

	def configure_basic(
		self,
		*,
		title: str,
		motion_name: str,
		robot_name: str,
		joint_names: list[str],
		joint_dims: list[int],
		link_names: list[str],
		imu_names: list[str],
		time_names: list[str],
		data_names: list[str],
		fps_by_section: dict[str, int] | None = None,
		initial_sections: dict[str, dict[str, Any]] | None = None,
	) -> None:
		self.configure(
			{
				"type": "init",
				"title": title,
				"motion_name": motion_name,
				"robot_name": robot_name,
				"joint_names": list(joint_names),
				"joint_dims": list(joint_dims),
				"link_names": list(link_names),
				"imu_names": list(imu_names),
				"time_names": list(time_names),
				"data_names": list(data_names),
				"fps_by_section": dict(fps_by_section or {}),
				"initial_sections": initial_sections or {},
			}
		)

	def enqueue_frame(self, payload: dict[str, Any]) -> None:
		self._ensure_thread_started()
		try:
			self.queue.put_nowait(payload)
		except queue.Full:
			try:
				self.queue.get_nowait()
			except queue.Empty:
				pass
			try:
				self.queue.put_nowait(payload)
			except queue.Full:
				pass

	def enqueue_omf_frame(self, frame_sections: dict[str, dict[str, Any]]) -> None:
		self.enqueue_frame({"sections": frame_sections})

	def close(self) -> None:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=1.0)
		self._disconnect()

	def _ensure_thread_started(self) -> None:
		if self._thread is not None and self._thread.is_alive():
			return
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._worker, daemon=True)
		self._thread.start()

	def _connect(self) -> bool:
		if self._init_payload is None:
			return False
		try:
			sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
			sock.settimeout(self.connect_timeout_s)
			_send_message(sock, self._init_payload)
			self._socket = sock
			print(f"{self.log_prefix} connected to {self.host}:{self.port}")
			return True
		except OSError as exc:
			now = time.monotonic()
			if now - self._last_connect_error_time > 5.0:
				print(f"{self.log_prefix} connect failed to {self.host}:{self.port}: {exc}")
				self._last_connect_error_time = now
			self._socket = None
			return False

	def _disconnect(self) -> None:
		if self._socket is not None:
			try:
				self._socket.close()
			except OSError:
				pass
			self._socket = None

	def _flush_batch(self, batch: list[dict[str, Any]]) -> bool:
		if not batch:
			return True
		if self._socket is None and not self._connect():
			return False
		try:
			assert self._socket is not None
			_send_message(self._socket, {"type": "frames", "frames": batch})
			return True
		except OSError:
			self._disconnect()
			return False

	def _worker(self) -> None:
		batch: list[dict[str, Any]] = []
		last_send_time = time.monotonic()
		while not self._stop_event.is_set():
			timeout = max(0.01, self.send_interval_s - (time.monotonic() - last_send_time))
			try:
				item = self.queue.get(timeout=timeout)
				batch.append(item)
			except queue.Empty:
				pass

			now = time.monotonic()
			should_send = bool(batch) and (
				len(batch) >= self.max_batch_size or (now - last_send_time) >= self.send_interval_s
			)
			if should_send:
				if self._flush_batch(batch):
					batch.clear()
					last_send_time = now
				else:
					time.sleep(min(1.0, self.send_interval_s))

		if batch:
			self._flush_batch(batch)
		if self._socket is not None:
			try:
				_send_message(self._socket, {"type": "close"})
			except OSError:
				pass
		self._disconnect()


class OMFStreamSession:
	def __init__(self) -> None:
		self.motion: OMF | None = None
		self.title = "OMF Live Viewer"
		self.live_data_names: list[str] = []

	def initialize(self, payload: dict[str, Any]) -> None:
		motion_name = str(payload.get("motion_name", "live_motion"))
		robot_name = str(payload.get("robot_name", "unknown_robot"))
		joint_names = list(payload.get("joint_names", []))
		joint_dims = list(payload.get("joint_dims", [1] * len(joint_names)))
		link_names = list(payload.get("link_names", []))
		imu_names = list(payload.get("imu_names", []))
		time_names = list(payload.get("time_names", ["state_fetch_time", "policy_run_time", "dt"]))
		data_names = list(payload.get("data_names", ["target", "model_target", "actual"]))
		fps_by_section = dict(payload.get("fps_by_section", {}))
		self.title = str(payload.get("title", motion_name or "OMF Live Viewer"))
		self.motion = OMF(
			name=motion_name,
			robot=robot_name,
			joint_names=joint_names,
			joint_dims=joint_dims,
			link_names=link_names,
			imu_names=imu_names,
			time_names=time_names,
			data_names=data_names,
		)
		for data_name, fps in fps_by_section.items():
			if data_name in self.motion.data_names:
				self.motion.data_section(data_name)["fps"] = int(fps or 0)
		initial_sections = payload.get("initial_sections", {})
		for data_name, section_data in initial_sections.items():
			if data_name in self.motion.data_names and isinstance(section_data, dict):
				section = self.motion.data_section(data_name)
				for key, value in section_data.items():
					section[key] = value
		self.motion.validate()
		self.live_data_names = [
			data_name for data_name in self.motion.data_names if data_name not in set(initial_sections.keys())
		]

	def append_frames(self, frames: list[dict[str, Any]], *, max_live_frames: int = 0) -> None:
		if self.motion is None:
			return
		for frame in frames:
			sections = frame.get("sections", {})
			if isinstance(sections, dict):
				self.motion.append_frame(sections, validate=False)
		if max_live_frames > 0 and self.live_data_names:
			self.motion.truncate_frames(max_live_frames, data_names=self.live_data_names, validate=False)


def _merge_channel_payloads(
	catalog_payload: dict[str, list[dict[str, Any]]],
	value_payload: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
	merged: dict[str, list[dict[str, Any]]] = {
		section_name: [dict(channel) for channel in channels]
		for section_name, channels in catalog_payload.items()
	}
	for section_name, channels in value_payload.items():
		by_key = {channel["key"]: channel for channel in merged.get(section_name, [])}
		section_channels = merged.setdefault(section_name, [])
		for channel in channels:
			existing = by_key.get(channel["key"])
			if existing is None:
				section_channels.append(dict(channel))
			else:
				existing["values"] = channel["values"]
				existing["x_values"] = channel["x_values"]
	return merged


class OMFStreamListener(threading.Thread):
	def __init__(self, host: str, port: int, event_queue: queue.Queue[dict[str, Any]]) -> None:
		super().__init__(daemon=True)
		self.host = host
		self.port = int(port)
		self.event_queue = event_queue
		self._stop_event = threading.Event()
		self._server_socket: socket.socket | None = None
		self._client_id_counter = count(1)
		self._client_threads: set[threading.Thread] = set()
		self._client_threads_lock = threading.Lock()

	def stop(self) -> None:
		self._stop_event.set()
		if self._server_socket is not None:
			try:
				self._server_socket.close()
			except OSError:
				pass
		with self._client_threads_lock:
			client_threads = list(self._client_threads)
		for client_thread in client_threads:
			client_thread.join(timeout=1.0)

	def _client_session_id(self, client_addr: tuple[str, int]) -> str:
		return f"client-{next(self._client_id_counter)}-{client_addr[0]}:{client_addr[1]}"

	def _put_client_event(self, client_id: str, client_addr: tuple[str, int], payload: dict[str, Any]) -> None:
		event = dict(payload)
		event["client_id"] = client_id
		event["client_addr"] = f"{client_addr[0]}:{client_addr[1]}"
		self.event_queue.put(event)

	def _handle_client(self, client_socket: socket.socket, client_addr: tuple[str, int]) -> None:
		client_id = self._client_session_id(client_addr)
		self._put_client_event(
			client_id,
			client_addr,
			{"type": "status", "message": f"client connected: {client_addr[0]}:{client_addr[1]}"},
		)
		with client_socket:
			client_socket.settimeout(2.0)
			while not self._stop_event.is_set():
				try:
					message = _recv_message(client_socket)
				except (OSError, ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError):
					break
				if message is None:
					break
				self._put_client_event(client_id, client_addr, message)
		self._put_client_event(
			client_id,
			client_addr,
			{"type": "status", "message": "client disconnected"},
		)
		self._put_client_event(client_id, client_addr, {"type": "disconnect"})

	def run(self) -> None:
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server_socket.bind((self.host, self.port))
		server_socket.listen()
		server_socket.settimeout(1.0)
		self._server_socket = server_socket
		try:
			while not self._stop_event.is_set():
				try:
					client_socket, client_addr = server_socket.accept()
				except socket.timeout:
					continue
				except OSError:
					break
				client_thread = threading.Thread(target=self._handle_client, args=(client_socket, client_addr), daemon=True)
				with self._client_threads_lock:
					self._client_threads.add(client_thread)
				client_thread.start()
				with self._client_threads_lock:
					finished_threads = {thread for thread in self._client_threads if not thread.is_alive()}
					self._client_threads.difference_update(finished_threads)
		finally:
			try:
				server_socket.close()
			except OSError:
				pass


def run_stream_listener(
	host: str = "0.0.0.0",
	port: int = 8765,
	refresh_interval_s: float = 0.02,
	rot_format: str | None = None,
	refresh_raw_data: bool = False,
	max_live_frames: int = 4000,
) -> int:
	try:
		from PySide6 import QtCore, QtWidgets
		from .viewer import OMFViewer, build_channel_specs
	except ImportError as exc:
		raise ImportError(
			"Streaming viewer requires optional dependencies. Install with: pip install 'otter-motion-format[viz]'"
		) from exc

	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
	app.setQuitOnLastWindowClosed(False)
	event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
	listener = OMFStreamListener(host=host, port=port, event_queue=event_queue)
	sessions: dict[str, OMFStreamSession] = {}
	viewers: dict[str, OMFViewer] = {}
	topology_signatures: dict[str, tuple[Any, ...]] = {}
	live_frame_limits: dict[str, int] = {}
	print(f"[otter-stream] listening on {host}:{port}")

	def handle_sigint(_signum, _frame) -> None:
		print("[otter-stream] received Ctrl+C, shutting down listener")
		app.quit()

	try:
		signal.signal(signal.SIGINT, handle_sigint)
	except ValueError:
		pass

	def _clear_viewer(client_id: str) -> None:
		viewers.pop(client_id, None)
		topology_signatures.pop(client_id, None)
		live_frame_limits.pop(client_id, None)

	def _current_max_live_frames(client_id: str) -> int:
		viewer = viewers.get(client_id)
		if viewer is not None:
			return int(viewer.max_live_frames)
		return int(live_frame_limits.get(client_id, max_live_frames))

	def _on_max_live_frames_changed(client_id: str, value: int) -> None:
		live_frame_limits[client_id] = max(0, int(value))
		session = sessions.get(client_id)
		if session is None or session.motion is None:
			return
		limit = live_frame_limits[client_id]
		if limit > 0 and session.live_data_names:
			session.motion.truncate_frames(limit, data_names=session.live_data_names, validate=False)
		refresh_viewer(client_id)

	def _selected_keys_for_viewer(client_id: str, session: OMFStreamSession) -> set[str]:
		viewer = viewers.get(client_id)
		if viewer is not None:
			checked = viewer.checked_channel_keys()
			if checked:
				return checked
		preselected = session.motion.resolve_chart_preselected(rot_format=rot_format) if session.motion is not None else {}
		return {key for keys in preselected.values() for key in keys}

	def refresh_viewer(client_id: str) -> None:
		session = sessions.get(client_id)
		if session is None or session.motion is None:
			return
		selected_keys = _selected_keys_for_viewer(client_id, session)
		value_payload = session.motion.build_chart_payload(
			rot_format=rot_format,
			channel_keys=selected_keys or None,
			include_values=True,
			validate=False,
		)
		viewer = viewers.get(client_id)
		signature = session.motion.chart_topology_signature(rot_format=rot_format)
		topology_changed = topology_signatures.get(client_id) != signature
		if viewer is None or topology_changed:
			catalog_payload = session.motion.build_chart_payload(
				rot_format=rot_format,
				include_values=False,
				validate=False,
			)
			sections = build_channel_specs(_merge_channel_payloads(catalog_payload, value_payload))
			preselected = session.motion.resolve_chart_preselected(rot_format=rot_format)
			raw_data = session.motion.to_dict() if refresh_raw_data else None
			if viewer is None:
				viewer = OMFViewer(
					title=session.title,
					sections=sections,
					preselected=preselected,
					layer_styles=session.motion._default_layer_styles(),
					raw_data=raw_data or {},
					max_live_frames=live_frame_limits.get(client_id, max_live_frames),
				)
				viewer.destroyed.connect(lambda *_args, client_id=client_id: _clear_viewer(client_id))
				viewer.maxLiveFramesChanged.connect(
					lambda value, client_id=client_id: _on_max_live_frames_changed(client_id, value)
				)
				viewer.liveSelectionChanged.connect(lambda client_id=client_id: refresh_viewer(client_id))
				viewers[client_id] = viewer
				live_frame_limits[client_id] = viewer.max_live_frames
				viewer.show()
				viewer.raise_()
				viewer.activateWindow()
				print(f"[otter-stream] viewer ready for {client_id}: {session.title}")
			else:
				viewer.set_sections(sections=sections, raw_data=raw_data, preselected=None)
				viewer.setWindowTitle(session.title)
			topology_signatures[client_id] = signature
			return

		viewer.update_live_channel_values(build_channel_specs(value_payload))
		viewer.setWindowTitle(session.title)

	def drain_queue() -> None:
		changed_clients: set[str] = set()
		while True:
			try:
				message = event_queue.get_nowait()
			except queue.Empty:
				break
			message_type = str(message.get("type", ""))
			client_id = str(message.get("client_id", "default"))
			client_addr = str(message.get("client_addr", "unknown"))
			if message_type == "init":
				session = sessions.setdefault(client_id, OMFStreamSession())
				print(
					f"[otter-stream] received init for {client_id} ({client_addr}): "
					f"{message.get('motion_name', 'live_motion')}"
				)
				session.initialize(message)
				topology_signatures.pop(client_id, None)
				changed_clients.add(client_id)
			elif message_type == "frames":
				session = sessions.get(client_id)
				if session is None:
					continue
				session.append_frames(
					list(message.get("frames", [])),
					max_live_frames=_current_max_live_frames(client_id),
				)
				changed_clients.add(client_id)
			elif message_type == "status":
				print(f"[otter-stream] {client_id} {message.get('message', '')}")
			elif message_type == "close":
				print(f"[otter-stream] stream closed by {client_id}")
			elif message_type == "disconnect":
				print(f"[otter-stream] connection ended for {client_id}")
		for changed_client_id in changed_clients:
			refresh_viewer(changed_client_id)

	listener.start()
	timer = QtCore.QTimer()
	timer.timeout.connect(drain_queue)
	timer.start(max(10, int(refresh_interval_s * 1000)))
	signal_timer = QtCore.QTimer()
	signal_timer.timeout.connect(lambda: None)
	signal_timer.start(100)

	try:
		return app.exec()
	finally:
		listener.stop()
		listener.join(timeout=1.0)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="otter-omf-listen",
		description="Listen for remote OMF live-stream data and display it in the otter viewer.",
	)
	parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind the listener to")
	parser.add_argument("--port", type=int, default=8765, help="TCP port for the listener")
	parser.add_argument(
		"--refresh-interval",
		type=float,
		default=0.02,
		help="GUI refresh interval in seconds (default: 0.02 for ~50 Hz)",
	)
	parser.add_argument(
		"--rot-format",
		choices=["rotvec", "euler", "both"],
		default="both",
		help="Rotation display format used by the viewer",
	)
	parser.add_argument(
		"--refresh-raw-data",
		action="store_true",
		help="Refresh the raw-data tree on each GUI update",
	)
	parser.add_argument(
		"--max-live-frames",
		type=int,
		default=4000,
		help="Initial rolling history window for live sections; 0 keeps all frames. Also configurable in the viewer UI.",
	)
	return parser


def listen_main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return run_stream_listener(
		host=args.host,
		port=args.port,
		refresh_interval_s=args.refresh_interval,
		rot_format=None if args.rot_format == "both" else args.rot_format,
		refresh_raw_data=args.refresh_raw_data,
		max_live_frames=args.max_live_frames,
	)


if __name__ == "__main__":
	raise SystemExit(listen_main())
