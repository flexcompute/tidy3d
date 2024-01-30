import ssl

from tidy3d.web.core.environment import Env


def test_tidy3d_cli():
    Env.set_ssl_version(ssl.TLSVersion.TLSv1_3)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_3

    Env.set_ssl_version(ssl.TLSVersion.TLSv1_2)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_2

    Env.set_ssl_version(ssl.TLSVersion.TLSv1_1)
    assert Env.current.ssl_version == ssl.TLSVersion.TLSv1_1

    Env.set_ssl_version(None)
    assert Env.current.ssl_version is None
