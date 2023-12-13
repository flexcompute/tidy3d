from tidy3d.web.core.task_info import RunInfo


def test_run_info_display():
    ri = RunInfo(perc_done=50, field_decay=1e-3)
    ri.display()
