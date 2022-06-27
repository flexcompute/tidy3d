from tidy3d.components.data.batch_data import BatchData

from .test_data_sim import make_sim_data

# monitor data instances

SIM_DATA = make_sim_data()
SIM_DATA_DICT = {f"sim_data_{i}": SIM_DATA for i in range(5)}


def make_batch_data():
    return BatchData(sim_data_dict=SIM_DATA_DICT)


def test_batch_data():
    data = make_batch_data()
