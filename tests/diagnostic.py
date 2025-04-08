import os
from datetime import datetime

import requests
from tidy3d import web
from tidy3d.web.environment import Env

s3_small_file_url = "https://s3.us-gov-west-1.amazonaws.com/www.simulation.cloud/test.txt"
s3_big_file_url = "https://s3.us-gov-west-1.amazonaws.com/www.simulation.cloud/50M.dat"
time_url = "http://worldtimeapi.org/api/timezone/Etc/UTC"
proxy_check_url = "https://httpbin.org/ip"
flexcompute_robots_url = "https://www.flexcompute.com/robots.txt"
simcloud_robots_url = "https://tidy3d.simulation.cloud/robots.txt"


def download_s3_file_with_boto3(object_key):
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config

        region_name = "us-gov-west-1"
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=region_name)
        s3.download_file("www.simulation.cloud", object_key, f"/tmp/{object_key}")
        print(f"download {object_key} with boto3 successfully")
    except Exception as e:
        print(f"download {object_key} with boto3 fails: {e}")


def download_file(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("downloaded_file", "wb") as f:
                f.write(response.content)
            print(f"download {url} with request successfully")
        else:
            print(f"download {url} with request fails, status_code: {response.status_code}")
    except Exception as e:
        print(f"download {url} with request error: {e}")


def get_file_content(url):
    try:
        response = requests.get(url)
        content = response.text
        return content
    except Exception as e:
        print(f"get_file_content {url} error: {e}")


def get_urllib_version():
    try:
        import urllib3

        return urllib3.__version__
    except Exception as e:
        print(f"get_urllib_version, error: {e}")


def get_boto_version():
    try:
        import importlib.metadata

        return importlib.metadata.version("boto3")
    except Exception as e:
        print(f"get_boto_version, error: {e}")


def list_ssl_info():
    try:
        import ssl

        context = ssl.create_default_context()
        ciphers = context.get_ciphers()
        for cipher in ciphers:
            print(
                f"Cipher: {cipher['name']}, Protocol: {cipher['protocol']}, Strength: {cipher['strength_bits']} bits"
            )
    except Exception as e:
        print(f"list_supported_algorithms,error: {e}")


def get_ssl_version():
    try:
        import ssl

        openssl_version = ssl.OPENSSL_VERSION
        print(f"OpenSSL Version: {openssl_version}")

        openssl_version_number = ssl.OPENSSL_VERSION_NUMBER
        print(f"OpenSSL Version Number: {openssl_version_number}")

        print(f"OpenSSL TLS 1.1: {ssl.HAS_TLSv1_1}")
        print(f"OpenSSL TLS 1.2: {ssl.HAS_TLSv1_2}")
        print(f"OpenSSL TLS 1.3: {ssl.HAS_TLSv1_3}")
    except Exception as e:
        print(f"get_ssl_version,error: {e}")


def detect_proxy_with_requests():
    proxies = {"http": os.getenv("HTTP_PROXY"), "https": os.getenv("HTTPS_PROXY")}
    try:
        response = requests.get(proxy_check_url, proxies=proxies, timeout=5)
        data = response.json()
        return data
    except Exception as e:
        print("detect_proxy_with_requests: error", e)


def get_network_time():
    try:
        response = requests.get(time_url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
        data = response.json()
        datetime_str = data.get("datetime")
        if datetime_str:
            return datetime_str
    except Exception as e:
        print("get_network_time error:{e}")


def get_basic_os_info():
    info = {}
    info["os_name"] = os.name  # 'posix', 'nt', etc.
    info["os_version"] = os.environ  # 'posix', 'nt', etc.
    info["env_variables"] = dict(os.environ)  # Environment variables
    info["current_directory"] = os.getcwd()  # Current working directory
    info["process_id"] = os.getpid()  # Current process ID
    try:
        info["user"] = os.getlogin()
    except Exception as e:
        info["user"] = str(e)  # Fallback for environments where getlogin() may fail
    return info


if __name__ == "__main__":
    print("---------Attention: Please install tidy3d first.----------\n")
    print("---------web.test information begin----------")
    response = requests.get(Env.current.web_api_endpoint)
    print(f"call endpoint response:{response.text}")
    print("---------web.test information end----------\n")

    print("---------web.test information begin----------")
    web.test()
    print("---------web.test information end----------\n")

    print("---------sdk information begin----------")
    version = web.__version__
    ssl_version = Env.current.ssl_version
    ssl_verify = Env.current.ssl_verify
    boto_version = get_boto_version()
    urllib_version = get_urllib_version()
    print(
        f"Tidy3d version: {version}, ssl_version: {ssl_version}, ssl_verify: {ssl_verify}, boto_version:{boto_version}, urllib_version:{urllib_version}"
    )
    print("---------sdk information end----------\n")

    print("---------ssl information begin----------")
    get_ssl_version()
    list_ssl_info()
    print("---------ssl information end----------\n")

    print("---------os information begin----------")
    os_info = get_basic_os_info()
    for key, value in os_info.items():
        print(f"{key}: {value}")
    print("---------os information end----------\n")

    print("---------get s3 public file information end----------")
    file_content = get_file_content(s3_small_file_url)
    print(f"get test.text file content: {file_content}")
    print(f"start to download {s3_small_file_url} with request...")
    download_file(s3_small_file_url)
    print(f"start to download {s3_small_file_url} with boto3...")
    download_s3_file_with_boto3("test.txt")
    print(f"start to download {s3_big_file_url} with request...")
    download_file(s3_big_file_url)
    print(f"start to download {s3_big_file_url} with boto3...")
    download_s3_file_with_boto3("50M.dat")
    print("---------get s3 public information end----------\n")

    print("---------time information begin----------")
    network_time = get_network_time()
    local_time = datetime.utcnow()
    print(f"local_time: {local_time}, network_time: {network_time}")
    print("---------time information end----------\n")

    print("---------network proxy information begin----------")
    proxy = detect_proxy_with_requests()
    print(f"proxy info: {proxy}")
    print("---------network proxy information end----------\n")

    print("---------download robots information begin----------")
    print(f"start download {flexcompute_robots_url}")
    download_file(flexcompute_robots_url)
    print(f"start download {simcloud_robots_url}")
    download_file(simcloud_robots_url)
    print("---------download robots information end----------\n")
