import tidy3d.web as web
from tidy3d import *
from tidy3d.web.task import TaskStatus

from tests.utils import clear_dir
from tests.utils import SIM_FULL as sim

# sim = {f"sim_{i}": sim for i in range(10)}
job = web.Job(simulation=sim, task_name='test')
job.upload()
job.get_info()
job.run()
job.monitor()