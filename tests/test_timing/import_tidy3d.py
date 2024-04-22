def test_import():
    import subprocess
    import time

    n = 100
    python_load_time = 0
    tidy3d_load_time = 0

    for _ in range(n):
        s = time.time()
        subprocess.call(["python", "-c", "import tidy3d"])
        tidy3d_load_time += time.time() - s

        s = time.time()
        subprocess.call(["python", "-c", "pass"])
        python_load_time += time.time() - s

    print(f"average tidy3d load time = {(tidy3d_load_time - python_load_time) / n}")


if __name__ == "__main__":
    test_import()
