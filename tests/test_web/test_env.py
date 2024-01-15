from tidy3d.web.core.environment import Env


def test_tidy3d_env():
    Env.enable_caching(True)
    assert Env.current.enable_caching == True

    Env.enable_caching(False)
    assert Env.current.enable_caching == False
