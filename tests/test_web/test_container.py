# """tests containers (Job, Batch)"""
# import pytest

# from .test_webapi import make_sim, set_api_key
# from .test_webapi import mock_upload, mock_get_info, mock_start, mock_download, mock_load
# from tidy3d.web.container import Job, Batch

# sim = make_sim()
# sims = dict(test=sim)

# @pytest.fixture
# def mock_webapi(set_api_key, mock_upload, mock_get_info, mock_start, mock_download, mock_load):
#     """Mocks all webapi operation."""

# def test_job(mock_webapi):

# 	j = Job(simulation=sim, task_name='test')
# 	sim_data = j.run()
