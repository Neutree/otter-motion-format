import os
import tempfile
import numpy as np
import otter_motion_format as omf
from robot_data_editor.omf_motion import load_motion_bundle
from robot_data_editor.sequence_editor import SequenceEditorWidget
from otter_motion_format.viewer import OMFViewer
from PyQt5.QtWidgets import QApplication
import sys

def test():
    try:
        # 1) Create OMF object
        motion = omf.OMF(name='test_motion', robot='agibot_x2', data_names=['target', 'actual'])
        l_target = motion.get_layer('target')
        l_target.fps = 30
        l_target.root_pos = np.zeros((3, 3))
        l_target.root_rot = np.zeros((3, 4))
        l_target.joint.pos = np.zeros((3, 1))
        l_target.joint.names = ['joint1']
        
        l_actual = motion.get_layer('actual')
        l_actual.fps = 60
        l_actual.root_pos = np.zeros((5, 3))
        l_actual.root_rot = np.zeros((5, 4))
        l_actual.joint.pos = np.zeros((5, 1))
        l_actual.joint.names = ['joint1']

        # 2) Save and load with resample
        with tempfile.NamedTemporaryFile(suffix='.msgpack', delete=False) as tmp:
            tmp_path = tmp.name
        motion.save(tmp_path)
        
        loaded = omf.load(tmp_path, resample_fps='max')
        assert loaded.get_layer('target').fps == 60, f"Target FPS expected 60, got {loaded.get_layer('target').fps}"
        assert loaded.get_layer('actual').fps == 60
        print("PASS: resample_fps='max'")

        # 3) Viewer sections and OMFViewer
        app = QApplication.instance() or QApplication(sys.argv)
        sections = []
        for name in loaded.data_names:
            sections.extend(loaded.get_layer(name)._build_channels(legend_prefix=f"{name}: "))
        
        viewer = OMFViewer(title='t', sections=sections, preselected=['target: root_pos_x'], raw_data=loaded.to_dict())
        
        # Assert splitter count is 2 (plot area and control panel)
        assert viewer.splitter.count() == 2, f"Splitter count expected 2, got {viewer.splitter.count()}"
        
        # Raw toggle switches stack index
        initial_index = viewer.stack.currentIndex()
        viewer.btn_raw.click()
        assert viewer.stack.currentIndex() != initial_index, "Stack index did not change after raw toggle"
        viewer.btn_raw.click()
        assert viewer.stack.currentIndex() == initial_index
        
        # Value column width
        # The table widget for a tab. Accessing via viewer.tabs.currentWidget().table
        table = viewer.tabs.widget(0).table
        assert table.columnWidth(1) == 112, f"Column width expected 112, got {table.columnWidth(1)}"
        
        # Refresh plot range stability
        viewer.plot_widget.setXRange(0, 1)
        viewer.plot_widget.setYRange(0, 1)
        viewer._refresh_plot()
        xr = viewer.plot_widget.viewRange()[0]
        yr = viewer.plot_widget.viewRange()[1]
        assert np.isclose(xr[0], 0, atol=0.1) and np.isclose(xr[1], 1, atol=0.1)
        assert np.isclose(yr[0], 0, atol=0.1) and np.isclose(yr[1], 1, atol=0.1)
        print("PASS: OMFViewer assertions")

        # 4) load_motion_bundle
        bundle = load_motion_bundle(tmp_path, resample_fps='max')
        assert 'target' in bundle.resampled_layers
        assert bundle.resampled_fps == 60
        print("PASS: load_motion_bundle")

        # 5) SequenceEditorWidget
        editor = SequenceEditorWidget()
        editor.set_motion(bundle.motion, bundle.reference_motion, bundle.overlay_motions)
        
        mapped = editor._mapped_overlay_series('target', 'joint::joint1')
        expected = bundle.overlay_motions['target'].channel_series('joint::joint1')
        assert np.allclose(mapped, expected), "Mapped overlay series mismatch"
        print("PASS: SequenceEditorWidget")
        
        os.remove(tmp_path)
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    test()
