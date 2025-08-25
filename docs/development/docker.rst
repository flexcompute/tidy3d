Docker Development Image Installation
=====================================


Quick Start Guide
-----------------

This guide is for anyone setting up the development environment for the first time or migrating from a version after ``2.9.0`` that the dev.Dockerfile image was enabled.

Prerequisites
^^^^^^^^^^^^^^^

1.  Install **Docker Engine**. You can find the official instructions here: `https://docs.docker.com/engine/install/ <https://docs.docker.com/engine/install/>`_.
    -   *Note for Ubuntu users*: It's recommended to avoid installing Docker from the Snap store.
2.  Follow the official `Linux post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_ to run Docker commands without ``sudo`` and to enable the Docker daemon to start on boot.
3.  **Reboot** your system for all group changes to take effect.

Setup and Usage
^^^^^^^^^^^^^^^^^

Follow these steps in your terminal to clone the repository, build the Docker image, and launch a development session.

.. code-block:: bash

    # 1. Clone the repository and navigate into it
    git clone https://github.com/flexcompute/tidy3d.git
    cd tidy3d

    # 2. Build the development Docker image
    docker build -t tidy3d-python-client-dev -f dev.Dockerfile .

    # 3. Create a persistent container with your local code mounted
    # This also maps port 8888 for tools like Jupyter Lab
    docker container create --name=tidy3d_python_client_dev \
      --userns=host \
      -p 8888:8888 \
      -v .:/home/flexdaemon/tidy3d \
      tidy3d-python-client-dev

    # 4. Start the container
    docker start tidy3d_python_client_dev

    # 5. Open an interactive shell inside the running container
    docker exec -it tidy3d_python_client_dev /bin/bash

You are now inside the container's shell. From here, you can set up the Python environment.

.. code-block:: bash

    # (Inside Container) 6. Create a virtual environment and activate it
    uv venv -p 3.11
    source .venv/bin/activate

    # (Inside Container) 7. Install the project in editable mode with dev dependencies
    uv pip install -e .[dev]

With the environment ready, you can run tests, format code, or start a Jupyter Lab session.

Running Jupyter Lab
^^^^^^^^^^^^^^^^^^^

To run a Jupyter Lab instance that's accessible from your host machine's web browser, execute the following command from within the container's shell:

.. code-block:: bash

    # (Inside Container) Make sure your virtual environment is activated!
    jupyter lab . --ip=0.0.0.0

After it starts, Jupyter will print a URL to the terminal containing a security token. It will look like this:
``http://127.0.0.1:8888/lab?token=...``

Copy this complete URL and paste it into your web browser (like Firefox) on your host machine to begin your session.
