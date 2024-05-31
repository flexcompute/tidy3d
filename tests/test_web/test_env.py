import ssl

from tidy3d.web.core.environment import Env


def test_tidy3d_env():
    Env.enable_caching(True)
    assert Env.current.enable_caching is True

    Env.enable_caching(False)
    assert Env.current.enable_caching is False


def test_set_ssl_version():
    Env.set_ssl_version(ssl.TLSVersion.TLSv1_3)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_3

    Env.set_ssl_version(ssl.TLSVersion.TLSv1_2)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_2

    Env.set_ssl_version(ssl.TLSVersion.TLSv1_1)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_1

    Env.set_ssl_version(None)
    assert Env.current.ssl_version is None
