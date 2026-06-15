from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(slots=True)
class ChannelSpec:
	key: str
	label: str
	values: np.ndarray
	x_values: np.ndarray
	layer_name: str


@dataclass(slots=True)
class LayerStyleSpec:
	line_style: str = "solid"
	opacity: float = 1.0
	width: float = 2.0


def build_channel_specs(section_payload: dict[str, list[dict[str, object]]]) -> dict[str, list[ChannelSpec]]:
	sections: dict[str, list[ChannelSpec]] = {}
	for section_name, channels in section_payload.items():
		sections[section_name] = [
			ChannelSpec(
				key=str(channel["key"]),
				label=str(channel["label"]),
				values=np.asarray(channel["values"], dtype=np.float64),
				x_values=np.asarray(channel["x_values"], dtype=np.float64),
				layer_name=section_name,
			)
			for channel in channels
		]
	return sections


class ColorCellButton(QtWidgets.QPushButton):
	colorChanged = QtCore.Signal(str)

	def __init__(self, color: str, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self._color = color
		self.clicked.connect(self._choose_color)
		self.setFixedWidth(34)
		self._apply_style()

	@property
	def color(self) -> str:
		return self._color

	def set_color(self, color: str) -> None:
		self._color = color
		self._apply_style()

	def _choose_color(self) -> None:
		color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color), self.window(), "Select curve color")
		if not color.isValid():
			return
		self._color = color.name()
		self._apply_style()
		self.colorChanged.emit(self._color)

	def _apply_style(self) -> None:
		self.setStyleSheet(f"QPushButton {{ background-color: {self._color}; border: 1px solid #52606d; border-radius: 4px; }}")


class ChannelListWidget(QtWidgets.QWidget):
	selectionChanged = QtCore.Signal()
	currentKeyChanged = QtCore.Signal(object)
	channelColorChanged = QtCore.Signal(str, str)

	VALUE_COLUMN = 0
	COLOR_COLUMN = 1
	NAME_COLUMN = 2
	VALUE_COLUMN_WIDTH = 112

	def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self._row_by_key: dict[str, int] = {}
		self._key_by_row: dict[int, str] = {}
		self._base_labels: dict[str, str] = {}
		self._value_items: dict[str, QtWidgets.QTableWidgetItem] = {}
		self._name_items: dict[str, QtWidgets.QTableWidgetItem] = {}
		self._color_buttons: dict[str, ColorCellButton] = {}
		self._suspend_item_signal = False
		self._value_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		header = QtWidgets.QHBoxLayout()
		layout.addLayout(header)
		header.addWidget(QtWidgets.QLabel(title))
		header.addStretch(1)

		self.search = QtWidgets.QLineEdit()
		self.search.setPlaceholderText("Search channels")
		layout.addWidget(self.search)

		button_row = QtWidgets.QHBoxLayout()
		layout.addLayout(button_row)
		self.select_all_button = QtWidgets.QPushButton("All")
		self.select_none_button = QtWidgets.QPushButton("None")
		self.select_invert_button = QtWidgets.QPushButton("Invert")
		button_row.addWidget(self.select_all_button)
		button_row.addWidget(self.select_none_button)
		button_row.addWidget(self.select_invert_button)

		self.table = QtWidgets.QTableWidget(0, 3)
		self.table.setHorizontalHeaderLabels(["Value", "Color", "Channel"])
		self.table.verticalHeader().setVisible(False)
		self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
		self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.table.setAlternatingRowColors(True)
		self.table.horizontalHeader().setStretchLastSection(True)
		self.table.horizontalHeader().setSectionResizeMode(self.VALUE_COLUMN, QtWidgets.QHeaderView.Fixed)
		self.table.horizontalHeader().setSectionResizeMode(self.COLOR_COLUMN, QtWidgets.QHeaderView.ResizeToContents)
		self.table.horizontalHeader().setSectionResizeMode(self.NAME_COLUMN, QtWidgets.QHeaderView.Stretch)
		self.table.setColumnWidth(self.VALUE_COLUMN, self.VALUE_COLUMN_WIDTH)
		layout.addWidget(self.table, 1)

		self.search.textChanged.connect(self._apply_filter)
		self.table.itemChanged.connect(self._on_item_changed)
		self.table.currentCellChanged.connect(self._on_current_cell_changed)
		self.select_all_button.clicked.connect(self._select_all)
		self.select_none_button.clicked.connect(self._select_none)
		self.select_invert_button.clicked.connect(self._select_invert)

	def set_channels(self, channels: list[ChannelSpec], preselected: set[str] | None = None, colors: dict[str, str] | None = None) -> None:
		selected = preselected or set()
		colors = colors or {}
		self._suspend_item_signal = True
		self.table.clearContents()
		self.table.setRowCount(len(channels))
		self._row_by_key.clear()
		self._key_by_row.clear()
		self._base_labels.clear()
		self._value_items.clear()
		self._name_items.clear()
		self._color_buttons.clear()
		for row, channel in enumerate(channels):
			self._row_by_key[channel.key] = row
			self._key_by_row[row] = channel.key
			self._base_labels[channel.key] = channel.label

			value_item = QtWidgets.QTableWidgetItem("")
			value_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
			value_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
			value_item.setFont(self._value_font)
			self.table.setItem(row, self.VALUE_COLUMN, value_item)
			self._value_items[channel.key] = value_item

			color_button = ColorCellButton(colors.get(channel.key, "#8ecae6"))
			color_button.colorChanged.connect(lambda color, key=channel.key: self.channelColorChanged.emit(key, color))
			self.table.setCellWidget(row, self.COLOR_COLUMN, color_button)
			self._color_buttons[channel.key] = color_button

			name_item = QtWidgets.QTableWidgetItem(channel.label)
			name_item.setData(QtCore.Qt.UserRole, channel.key)
			name_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable)
			name_item.setCheckState(QtCore.Qt.Checked if channel.key in selected else QtCore.Qt.Unchecked)
			self.table.setItem(row, self.NAME_COLUMN, name_item)
			self._name_items[channel.key] = name_item
			self.table.setRowHeight(row, 24)
		self._suspend_item_signal = False
		if channels:
			preferred_key = next(iter(selected), channels[0].key if channels else None)
			if preferred_key is not None:
				self._select_key(preferred_key)
		self._apply_filter(self.search.text())
		self.selectionChanged.emit()

	def checked_keys(self) -> list[str]:
		return [key for key, item in self._name_items.items() if item.checkState() == QtCore.Qt.Checked]

	def current_key(self) -> str | None:
		row = self.table.currentRow()
		return self._key_by_row.get(row)

	def channel_color(self, key: str) -> str:
		button = self._color_buttons.get(key)
		return button.color if button is not None else "#8ecae6"

	def clear_values(self) -> None:
		for item in self._value_items.values():
			item.setText("")

	def set_value_text(self, key: str, text: str) -> None:
		item = self._value_items.get(key)
		if item is not None:
			item.setText(text)

	def _apply_filter(self, text: str) -> None:
		needle = text.strip().lower()
		for key, row in self._row_by_key.items():
			label = self._base_labels.get(key, "")
			self.table.setRowHidden(row, bool(needle) and needle not in label.lower())

	def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
		if self._suspend_item_signal or item.column() != self.NAME_COLUMN:
			return
		self.selectionChanged.emit()

	def _on_current_cell_changed(self, current_row: int, _current_column: int, _previous_row: int, _previous_column: int) -> None:
		key = self._key_by_row.get(current_row)
		if key is None:
			self.currentKeyChanged.emit(None)
			return
		item = self._name_items[key]
		if item.checkState() != QtCore.Qt.Checked:
			self._suspend_item_signal = True
			item.setCheckState(QtCore.Qt.Checked)
			self._suspend_item_signal = False
			self.selectionChanged.emit()
		self.currentKeyChanged.emit(key)

	def _select_key(self, key: str) -> None:
		row = self._row_by_key.get(key)
		if row is None:
			return
		self.table.setCurrentCell(row, self.NAME_COLUMN)

	def _set_visible_items(self, state_getter) -> None:
		self._suspend_item_signal = True
		for key, row in self._row_by_key.items():
			if self.table.isRowHidden(row):
				continue
			self._name_items[key].setCheckState(state_getter(self._name_items[key]))
		self._suspend_item_signal = False
		self.selectionChanged.emit()

	def _select_all(self) -> None:
		self._set_visible_items(lambda _item: QtCore.Qt.Checked)

	def _select_none(self) -> None:
		self._set_visible_items(lambda _item: QtCore.Qt.Unchecked)

	def _select_invert(self) -> None:
		self._set_visible_items(lambda item: QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked)


class OMFViewer(QtWidgets.QWidget):
	_STYLE_MAP = {
		"solid": QtCore.Qt.SolidLine,
		"dash": QtCore.Qt.DashLine,
		"dot": QtCore.Qt.DotLine,
		"dashdot": QtCore.Qt.DashDotLine,
	}

	def __init__(
		self,
		title: str,
		sections: dict[str, list[ChannelSpec]],
		preselected: dict[str, set[str]] | None = None,
		layer_styles: dict[str, LayerStyleSpec] | dict[str, dict[str, object]] | None = None,
		raw_data: dict[str, object] | None = None,
		parent: QtWidgets.QWidget | None = None,
	) -> None:
		super().__init__(parent)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
		self.setWindowTitle(title)
		self.resize(1440, 900)
		self._curves: dict[str, pg.PlotDataItem] = {}
		self._active_keys: dict[str, str | None] = {}
		self._visible_channels: list[ChannelSpec] = []
		self._channel_maps = {
			section_name: {channel.key: channel for channel in section_channels}
			for section_name, section_channels in sections.items()
		}
		self._channel_lists: dict[str, ChannelListWidget] = {}
		self._layer_styles = self._normalize_layer_styles(layer_styles or {}, list(sections.keys()))
		self._channel_colors: dict[str, str] = {}
		self._raw_data = raw_data or {}
		self._has_rendered_plot = False
		self._hover_detail_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
		self._follow_latest_enabled = False
		self._latest_x_value: float | None = None

		pg.setConfigOptions(antialias=False)

		root_layout = QtWidgets.QHBoxLayout(self)
		root_layout.setContentsMargins(8, 8, 8, 8)
		root_layout.setSpacing(8)

		self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		self.splitter.setChildrenCollapsible(False)
		root_layout.addWidget(self.splitter, 1)

		left_panel = QtWidgets.QWidget()
		left_layout = QtWidgets.QVBoxLayout(left_panel)
		left_layout.setContentsMargins(0, 0, 0, 0)
		left_layout.setSpacing(6)
		left_toolbar = QtWidgets.QHBoxLayout()
		left_layout.addLayout(left_toolbar)
		self.raw_toggle_button = QtWidgets.QPushButton("Raw Data")
		self.raw_toggle_button.setCheckable(True)
		self.raw_toggle_button.toggled.connect(self._toggle_raw_view)
		left_toolbar.addWidget(self.raw_toggle_button)
		self.follow_latest_button = QtWidgets.QPushButton("Follow Latest")
		self.follow_latest_button.setCheckable(True)
		self.follow_latest_button.setToolTip("Keep the newest frame anchored at its current on-screen position while live data grows.")
		self.follow_latest_button.toggled.connect(self._on_follow_latest_toggled)
		left_toolbar.addWidget(self.follow_latest_button)
		left_toolbar.addStretch(1)

		self.tabs = QtWidgets.QTabWidget()
		self.tabs.setMinimumWidth(360)
		left_layout.addWidget(self.tabs, 1)
		self._set_section_tabs(sections, preselected=preselected, preserve_state=False)

		right_panel = QtWidgets.QWidget()
		right_layout = QtWidgets.QVBoxLayout(right_panel)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.setSpacing(6)

		self.right_stack = QtWidgets.QStackedWidget()
		right_layout.addWidget(self.right_stack, 1)

		plot_page = QtWidgets.QWidget()
		plot_layout = QtWidgets.QVBoxLayout(plot_page)
		plot_layout.setContentsMargins(0, 0, 0, 0)
		plot_layout.setSpacing(6)

		help_label = QtWidgets.QLabel("Mouse drag to pan, wheel to zoom. Checked channels from all tabs are drawn together.")
		plot_layout.addWidget(help_label)

		self.hover_label = QtWidgets.QLabel("Move the mouse over the plot to inspect the current time.")
		self.hover_label.setMinimumHeight(24)
		plot_layout.addWidget(self.hover_label)

		self.plot = pg.PlotWidget()
		self.plot.setBackground("#0f1720")
		self.plot.showGrid(x=True, y=True, alpha=0.16)
		self.plot.setLabel("bottom", "Time (s)")
		self.plot.setLabel("left", "Value")
		self.plot.getPlotItem().vb.setMouseEnabled(x=True, y=True)
		self.hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f8fafc", width=1, style=QtCore.Qt.DashLine))
		self.hover_line.hide()
		self.plot.addItem(self.hover_line)
		plot_layout.addWidget(self.plot, 1)

		self.hover_detail_label = QtWidgets.QLabel("")
		self.hover_detail_label.setFont(self._hover_detail_font)
		self.hover_detail_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
		self.hover_detail_label.setWordWrap(True)
		self.hover_detail_label.setMinimumHeight(36)
		self.hover_detail_label.setMaximumHeight(60)
		self.hover_detail_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
		self.hover_detail_label.setStyleSheet("QLabel { background: #111827; color: #dbeafe; border: 1px solid #263244; padding: 6px; }")
		plot_layout.addWidget(self.hover_detail_label)

		self.raw_tree = QtWidgets.QTreeWidget()
		self.raw_tree.setColumnCount(2)
		self.raw_tree.setHeaderLabels(["Key", "Value"])
		self.raw_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		self.raw_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
		self.raw_tree.setAlternatingRowColors(True)
		self.raw_tree.setUniformRowHeights(True)
		self._populate_raw_tree()

		self.right_stack.addWidget(plot_page)
		self.right_stack.addWidget(self.raw_tree)
		self.right_stack.setCurrentWidget(plot_page)

		self.splitter.addWidget(left_panel)
		self.splitter.addWidget(right_panel)
		self.splitter.setStretchFactor(0, 0)
		self.splitter.setStretchFactor(1, 1)
		self.splitter.setSizes([420, 1020])
		self._hover_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)

		self.tabs.currentChanged.connect(self._refresh_plot)
		self._refresh_plot()

	def _on_follow_latest_toggled(self, checked: bool) -> None:
		self._follow_latest_enabled = bool(checked)

	def _set_status_text(self, x_value: float | None, layer_frames: dict[str, tuple[int, int]]) -> None:
		if x_value is not None:
			self.hover_label.setText(f"t = {x_value:.3f}s")
		elif not self.hover_label.text().strip():
			self.hover_label.setText("t = --")

		if layer_frames:
			detail_lines = [f"{layer_name} {frame_index}/{frame_count}" for layer_name, (frame_index, frame_count) in layer_frames.items()]
			self.hover_detail_label.setText(" | ".join(detail_lines))
		elif not self.hover_detail_label.text().strip():
			self.hover_detail_label.setText("frames: --")

	def _latest_layer_frames(self) -> dict[str, tuple[int, int]]:
		layer_frames: dict[str, tuple[int, int]] = {}
		for channel in self._visible_channels:
			if channel.x_values.size == 0:
				continue
			layer_frames[channel.layer_name] = (channel.x_values.size, channel.x_values.size)
		return layer_frames

	def _capture_ui_state(self) -> tuple[dict[str, set[str]], dict[str, str | None], int]:
		selected = {section_name: set(list_widget.checked_keys()) for section_name, list_widget in self._channel_lists.items()}
		active = {section_name: list_widget.current_key() for section_name, list_widget in self._channel_lists.items()}
		return selected, active, self.tabs.currentIndex()

	def _set_section_tabs(
		self,
		sections: dict[str, list[ChannelSpec]],
		preselected: dict[str, set[str]] | None = None,
		preserve_state: bool = True,
	) -> None:
		selected_state: dict[str, set[str]] = preselected or {}
		active_state: dict[str, str | None] = {}
		current_tab_index = 0
		if preserve_state and self._channel_lists:
			captured_selected, captured_active, current_tab_index = self._capture_ui_state()
			if preselected is None:
				selected_state = captured_selected
			active_state = captured_active

		self.tabs.blockSignals(True)
		while self.tabs.count() > 0:
			widget = self.tabs.widget(0)
			self.tabs.removeTab(0)
			if widget is not None:
				widget.deleteLater()

		self._channel_maps = {
			section_name: {channel.key: channel for channel in section_channels}
			for section_name, section_channels in sections.items()
		}
		self._channel_lists = {}
		self._active_keys = {}

		for section_name, section_channels in sections.items():
			for channel in section_channels:
				self._channel_colors.setdefault(channel.key, self._default_color_for_channel(channel.key))
			list_widget = ChannelListWidget(section_name)
			list_widget.set_channels(
				section_channels,
				preselected=selected_state.get(section_name),
				colors={channel.key: self._channel_colors[channel.key] for channel in section_channels},
			)
			list_widget.selectionChanged.connect(self._refresh_plot)
			list_widget.currentKeyChanged.connect(lambda key, section_name=section_name: self._on_current_key_changed(section_name, key))
			list_widget.channelColorChanged.connect(self._on_channel_color_changed)
			desired_active_key = active_state.get(section_name)
			if desired_active_key is not None:
				list_widget._select_key(desired_active_key)
			self._active_keys[section_name] = list_widget.current_key()
			self._channel_lists[section_name] = list_widget
			self.tabs.addTab(list_widget, section_name)

		if self.tabs.count() > 0:
			self.tabs.setCurrentIndex(max(0, min(current_tab_index, self.tabs.count() - 1)))
		self.tabs.blockSignals(False)

	def set_sections(
		self,
		sections: dict[str, list[ChannelSpec]],
		raw_data: dict[str, object] | None = None,
		preselected: dict[str, set[str]] | None = None,
	) -> None:
		current_keys = {
			section_name: list(channel_map.keys())
			for section_name, channel_map in self._channel_maps.items()
		}
		new_keys = {
			section_name: [channel.key for channel in section_channels]
			for section_name, section_channels in sections.items()
		}
		topology_changed = current_keys != new_keys or set(current_keys.keys()) != set(new_keys.keys())
		if topology_changed:
			self._set_section_tabs(sections, preselected=preselected, preserve_state=True)
		else:
			self._channel_maps = {
				section_name: {channel.key: channel for channel in section_channels}
				for section_name, section_channels in sections.items()
			}
		if raw_data is not None:
			self._raw_data = raw_data
			self._populate_raw_tree()
		if topology_changed or not self._refresh_plot_in_place():
			self._refresh_plot()

	def _toggle_raw_view(self, checked: bool) -> None:
		self.right_stack.setCurrentIndex(1 if checked else 0)
		self.raw_toggle_button.setText("Show Plot" if checked else "Raw Data")

	def _populate_raw_tree(self) -> None:
		self.raw_tree.clear()
		for key, value in self._raw_data.items():
			item = self._build_tree_item(key, value)
			item.setExpanded(True)
			self.raw_tree.addTopLevelItem(item)

	def _build_tree_item(self, key: str, value: object) -> QtWidgets.QTreeWidgetItem:
		item = QtWidgets.QTreeWidgetItem([str(key), self._format_tree_value(value)])
		if isinstance(value, dict):
			for child_key, child_value in value.items():
				item.addChild(self._build_tree_item(str(child_key), child_value))
		elif isinstance(value, list):
			for index, child_value in enumerate(value):
				item.addChild(self._build_tree_item(f"[{index}]", child_value))
		return item

	def _format_tree_value(self, value: object) -> str:
		if isinstance(value, dict):
			return f"{{{len(value)} items}}"
		if isinstance(value, list):
			return f"[{len(value)} items]"
		return str(value)

	def _normalize_layer_styles(
		self,
		layer_styles: dict[str, LayerStyleSpec] | dict[str, dict[str, object]],
		layer_names: list[str],
	) -> dict[str, LayerStyleSpec]:
		resolved: dict[str, LayerStyleSpec] = {}
		for layer_name in layer_names:
			style = layer_styles.get(layer_name)
			if isinstance(style, LayerStyleSpec):
				resolved[layer_name] = style
			elif isinstance(style, dict):
				resolved[layer_name] = LayerStyleSpec(
					line_style=str(style.get("line_style", "solid")),
					opacity=float(style.get("opacity", 1.0)),
					width=float(style.get("width", 2.0)),
				)
			else:
				resolved[layer_name] = LayerStyleSpec()
		return resolved

	def _default_color_for_channel(self, channel_key: str) -> str:
		palette = [
			"#4cc9f0",
			"#f72585",
			"#4361ee",
			"#ffd166",
			"#06d6a0",
			"#ef476f",
			"#8ecae6",
			"#fb8500",
			"#90be6d",
			"#b5179e",
			"#00b4d8",
			"#e76f51",
		]
		return palette[zlib.crc32(channel_key.encode("utf-8")) % len(palette)]

	def _selected_channels(self) -> list[ChannelSpec]:
		selected_channels: list[ChannelSpec] = []
		for section_name, list_widget in self._channel_lists.items():
			self._active_keys[section_name] = list_widget.current_key()
			channel_map = self._channel_maps[section_name]
			for key in list_widget.checked_keys():
				channel = channel_map.get(key)
				if channel is not None:
					selected_channels.append(channel)
		return selected_channels

	def _sorted_visible_channels(self, selected_channels: list[ChannelSpec]) -> list[ChannelSpec]:
		return sorted(
			selected_channels,
			key=lambda channel: (channel.key == self._active_keys.get(channel.layer_name), channel.label),
		)

	def _update_existing_curves(self, sorted_channels: list[ChannelSpec], view_range) -> bool:
		current_keys = list(self._curves.keys())
		sorted_keys = [channel.key for channel in sorted_channels]
		if current_keys != sorted_keys:
			return False

		follow_anchor_ratio: float | None = None
		follow_anchor_width: float | None = None
		if (
			self._follow_latest_enabled
			and view_range is not None
			and self._latest_x_value is not None
		):
			x_left, x_right = view_range[0]
			follow_anchor_width = x_right - x_left
			if follow_anchor_width > 1e-12:
				follow_anchor_ratio = (self._latest_x_value - x_left) / follow_anchor_width
				follow_anchor_ratio = min(1.0, max(0.0, follow_anchor_ratio))

		x_min = float("inf")
		x_max = float("-inf")
		for channel in sorted_channels:
			curve = self._curves.get(channel.key)
			if curve is None:
				return False
			style = self._layer_styles.get(channel.layer_name, LayerStyleSpec())
			color = self._channel_colors.get(channel.key, self._default_color_for_channel(channel.key))
			is_active = channel.key == self._active_keys.get(channel.layer_name)
			curve.setData(channel.x_values, channel.values)
			curve.setPen(self._build_pen(color, style, is_active))
			curve.setOpacity(min(1.0, style.opacity + 0.15) if is_active else max(0.24, style.opacity * 0.75))
			if channel.x_values.size > 0:
				x_min = min(x_min, float(channel.x_values[0]))
				x_max = max(x_max, float(channel.x_values[-1]))

		self._visible_channels = sorted_channels
		self._latest_x_value = x_max if np.isfinite(x_max) else None
		if (
			self._follow_latest_enabled
			and view_range is not None
			and follow_anchor_ratio is not None
			and follow_anchor_width is not None
			and np.isfinite(x_max)
		):
			new_x_left = x_max - follow_anchor_ratio * follow_anchor_width
			new_x_right = new_x_left + follow_anchor_width
			self.plot.setXRange(new_x_left, new_x_right, padding=0.0)
			self.plot.setYRange(view_range[1][0], view_range[1][1], padding=0.0)
		elif view_range is not None:
			self.plot.setXRange(view_range[0][0], view_range[0][1], padding=0.0)
			self.plot.setYRange(view_range[1][0], view_range[1][1], padding=0.0)
		elif np.isfinite(x_min) and np.isfinite(x_max):
			self.plot.setXRange(x_min, x_max if x_max > x_min else x_min + 1.0, padding=0.01)

		self._set_status_text(self._latest_x_value, self._latest_layer_frames())
		return True

	def _refresh_plot(self, *_args) -> None:
		preserve_view = self._has_rendered_plot and self.right_stack.currentIndex() == 0
		view_range = self.plot.getPlotItem().viewRange() if preserve_view else None
		follow_anchor_ratio: float | None = None
		follow_anchor_width: float | None = None
		if (
			self._follow_latest_enabled
			and preserve_view
			and view_range is not None
			and self._latest_x_value is not None
		):
			x_left, x_right = view_range[0]
			follow_anchor_width = x_right - x_left
			if follow_anchor_width > 1e-12:
				follow_anchor_ratio = (self._latest_x_value - x_left) / follow_anchor_width
				follow_anchor_ratio = min(1.0, max(0.0, follow_anchor_ratio))
		self.plot.clear()
		self.plot.addItem(self.hover_line)
		self.hover_line.hide()
		self._curves.clear()
		self._visible_channels.clear()
		for list_widget in self._channel_lists.values():
			list_widget.clear_values()

		selected_channels = self._selected_channels()
		if not selected_channels:
			self.hover_label.setText("No channels selected")
			self.hover_detail_label.setText("frames: --")
			return

		x_min = float("inf")
		x_max = float("-inf")
		sorted_channels = self._sorted_visible_channels(selected_channels)
		for channel in sorted_channels:
			style = self._layer_styles.get(channel.layer_name, LayerStyleSpec())
			color = self._channel_colors.get(channel.key, self._default_color_for_channel(channel.key))
			is_active = channel.key == self._active_keys.get(channel.layer_name)
			curve = self.plot.plot(channel.x_values, channel.values, pen=self._build_pen(color, style, is_active))
			curve.setClipToView(True)
			curve.setDownsampling(auto=True, method="peak")
			curve.setSkipFiniteCheck(True)
			curve.setOpacity(min(1.0, style.opacity + 0.15) if is_active else max(0.24, style.opacity * 0.75))
			self._curves[channel.key] = curve
			self._visible_channels.append(channel)
			if channel.x_values.size > 0:
				x_min = min(x_min, float(channel.x_values[0]))
				x_max = max(x_max, float(channel.x_values[-1]))
		self._latest_x_value = x_max if np.isfinite(x_max) else None
		if (
			self._follow_latest_enabled
			and preserve_view
			and view_range is not None
			and follow_anchor_ratio is not None
			and follow_anchor_width is not None
			and np.isfinite(x_max)
		):
			new_x_left = x_max - follow_anchor_ratio * follow_anchor_width
			new_x_right = new_x_left + follow_anchor_width
			self.plot.setXRange(new_x_left, new_x_right, padding=0.0)
			self.plot.setYRange(view_range[1][0], view_range[1][1], padding=0.0)
		elif preserve_view and view_range is not None:
			self.plot.setXRange(view_range[0][0], view_range[0][1], padding=0.0)
			self.plot.setYRange(view_range[1][0], view_range[1][1], padding=0.0)
		elif np.isfinite(x_min) and np.isfinite(x_max):
			self.plot.setXRange(x_min, x_max if x_max > x_min else x_min + 1.0, padding=0.01)
		self._set_status_text(self._latest_x_value, self._latest_layer_frames())
		self._has_rendered_plot = True

	def _refresh_plot_in_place(self) -> bool:
		if not self._has_rendered_plot or self.right_stack.currentIndex() != 0:
			return False
		for list_widget in self._channel_lists.values():
			list_widget.clear_values()
		selected_channels = self._selected_channels()
		if not selected_channels:
			self.hover_label.setText("No channels selected")
			self.hover_detail_label.setText("frames: --")
			return False
		return self._update_existing_curves(self._sorted_visible_channels(selected_channels), self.plot.getPlotItem().viewRange())

	def _build_pen(self, color: str, style: LayerStyleSpec, highlighted: bool) -> QtGui.QPen:
		return pg.mkPen(
			QtGui.QColor(color),
			width=style.width + (1.4 if highlighted else 0.0),
			style=self._STYLE_MAP.get(style.line_style, QtCore.Qt.SolidLine),
		)

	def _on_current_key_changed(self, section_name: str, key: str | None) -> None:
		self._active_keys[section_name] = key
		self._refresh_plot()

	def _on_channel_color_changed(self, key: str, color: str) -> None:
		self._channel_colors[key] = color
		self._refresh_plot()

	def _on_mouse_moved(self, event) -> None:
		if not self._visible_channels:
			self.hover_line.hide()
			for list_widget in self._channel_lists.values():
				list_widget.clear_values()
			return
		scene_pos = event[0]
		if not self.plot.sceneBoundingRect().contains(scene_pos):
			self.hover_line.hide()
			for list_widget in self._channel_lists.values():
				list_widget.clear_values()
			return
		mouse_point = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
		x_value = float(mouse_point.x())
		self.hover_line.setValue(x_value)
		self.hover_line.show()
		for list_widget in self._channel_lists.values():
			list_widget.clear_values()
		layer_frames: dict[str, tuple[int, int]] = {}
		for channel in self._visible_channels:
			if channel.x_values.size == 0:
				continue
			index = int(np.searchsorted(channel.x_values, x_value, side="left"))
			if index >= channel.x_values.size:
				index = channel.x_values.size - 1
			elif index > 0 and abs(channel.x_values[index - 1] - x_value) <= abs(channel.x_values[index] - x_value):
				index -= 1
			self._channel_lists[channel.layer_name].set_value_text(channel.key, f"{float(channel.values[index]):.6f}")
			layer_frames.setdefault(channel.layer_name, (index + 1, channel.x_values.size))
		self._set_status_text(x_value, layer_frames)


def show_omf_viewer(
	title: str,
	sections: dict[str, list[ChannelSpec]],
	preselected: dict[str, set[str]] | None = None,
	layer_styles: dict[str, LayerStyleSpec] | dict[str, dict[str, object]] | None = None,
	raw_data: dict[str, object] | None = None,
) -> None:
	app = QtWidgets.QApplication.instance()
	owns_app = app is None
	if app is None:
		app = QtWidgets.QApplication([])
	viewer = OMFViewer(title=title, sections=sections, preselected=preselected, layer_styles=layer_styles, raw_data=raw_data)
	viewer.show()
	viewer.raise_()
	viewer.activateWindow()
	if owns_app:
		app.exec()