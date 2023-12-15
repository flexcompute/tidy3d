from tidy3d.web.task import RunInfo


def test_run_info_display():
    ri = RunInfo(perc_done=50, field_decay=1e-3)
    ri.display()
