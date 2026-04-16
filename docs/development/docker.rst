Docker Development Environment
=============================

This guide will get you up and running with Tidy3D Python client development using Docker. The Docker workflow provides an isolated, reproducible development environment without affecting your system packages.

.. contents:: Quick Links
   :local:
   :depth: 2

Overview
--------

The Docker development environment provides:

* **Isolation**: Clean environment separate from your system
* **Reproducibility**: Everyone uses the same base environment
* **Simplicity**: No need to manage system dependencies
* **Flexibility**: Easy to reset and start fresh

**What's included:**

* Python 3.11 with ``uv`` package manager
* uv for dependency management
* AWS CLI for CodeArtifact access
* Neovim, git, pandoc for documentation
* Jupyter Lab for notebook development

Prerequisites
-------------

System Requirements
^^^^^^^^^^^^^^^^^^^

* **Operating System**: Ubuntu 22.04+, macOS, or Windows with WSL2
* **Docker Engine**: Installed and running
* **Git**: With SSH keys configured for GitHub
* **Disk Space**: ~2GB for Docker image + container

Quick Check
~~~~~~~~~~~

.. code-block:: bash

    # Check Docker is installed and running
    docker --version
    docker ps

    # Check Git SSH access
    ssh -T git@github.com

Installing Docker
^^^^^^^^^^^^^^^^^

If you don't have Docker installed:

**Ubuntu**

.. code-block:: bash

    # Install Docker Engine (NOT from snap)
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh

    # Add your user to docker group
    sudo usermod -aG docker $USER

    # Reboot to apply group changes
    sudo reboot

    # Verify installation
    docker run hello-world

See: `Docker Ubuntu Install Guide <https://docs.docker.com/engine/install/ubuntu/>`_

**macOS**

Download and install Docker Desktop from `docker.com <https://www.docker.com/products/docker-desktop>`_

**Windows**

Install Docker Desktop with WSL2 backend from `docker.com <https://www.docker.com/products/docker-desktop>`_

Access Requirements
^^^^^^^^^^^^^^^^^^^

* **GitHub**: Access to `flexcompute/tidy3d <https://github.com/flexcompute/tidy3d>`_
* **Tidy3D Account**: Sign up at `tidy3d.simulation.cloud <https://tidy3d.simulation.cloud/signup>`_
* **API Key**: Get yours from `account page <https://tidy3d.simulation.cloud/account?tab=apikey>`_

Quick Start Guide
-----------------

This guide is for anyone setting up the development environment for the first time or migrating from a version after ``2.9.0`` when the dev.Dockerfile image was enabled.

Step 1: Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Clone the repository and navigate into it
    git clone https://github.com/flexcompute/tidy3d.git
    cd tidy3d

    # Checkout develop branch or your feature branch
    git checkout develop

Step 2: Build the Docker Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build the development Docker image (~5 minutes):

.. code-block:: bash

    # Build the development Docker image
    docker build -t tidy3d-python-client-dev -f dev.Dockerfile .

The image includes:

* Debian base with Python 3.11
* ``uv`` for fast package installation
* uv for dependency management
* AWS CLI, git, pandoc, neovim
* ``flexdaemon`` user (UID 1000, GID 1000)

Step 3: Create the Development Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a persistent container with your local code mounted:

.. code-block:: bash

    # Create a persistent container with your local code mounted
    # This also maps port 8888 for tools like Jupyter Lab
    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -p 8888:8888 \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

**Flags explained:**

* ``--name``: Container name for easy reference
* ``--userns=host``: Use host user namespace (prevents permission issues)
* ``-p 8888:8888``: Expose port 8888 for Jupyter Lab
* ``-v .:/home/flexdaemon/tidy3d``: Mount current directory into container

Step 4: Start the Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Start the container
    docker start tidy3d_python_client_dev

Step 5: Open an Interactive Shell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Open an interactive shell inside the running container
    docker exec -it tidy3d_python_client_dev /bin/bash

You're now inside the container at ``/home/flexdaemon``!

Step 6: Set Up Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the container, create a virtual environment and install the package:

.. code-block:: bash

    # (Inside Container) Navigate to the mounted tidy3d directory
    cd tidy3d

    # (Inside Container) Create a virtual environment and activate it
    uv venv -p 3.11
    source .venv/bin/activate

    # (Inside Container) Install the project in editable mode with dev dependencies
    uv pip install -e .[dev]

.. note::
    The virtual environment is created inside the mounted directory, so it persists between container restarts.

Step 7: Configure API Key and Run Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure your API key and verify everything is working:

.. code-block:: bash

    # (Inside Container) Configure your API key
    tidy3d configure --apikey=YOUR_API_KEY_HERE

    # (Inside Container) Test the configuration
    python -c "import tidy3d.web as web; web.test()"

    # (Inside Container) Run tests
    pytest tests/ -v

If tests pass, you're ready to develop!

Development Workflow
--------------------

Daily Workflow
^^^^^^^^^^^^^^

1. **Start the container** (if not running):

   .. code-block:: bash

       # On host
       docker start tidy3d_python_client_dev
       docker exec -it tidy3d_python_client_dev bash

2. **Activate virtual environment** (inside container):

   .. code-block:: bash

       cd tidy3d
       source .venv/bin/activate

3. **Make changes** to the code on your host machine using your favorite editor (VSCode, PyCharm, etc.)

   Changes are immediately reflected in the container since the directory is mounted.

4. **Run tests** (inside container):

   .. code-block:: bash

       # Run all tests
       pytest

       # Run specific test file
       pytest tests/test_simulation.py

       # Run with verbose output
       pytest -v -s

5. **Commit changes** (on host or in container):

   .. code-block:: bash

       git add .
       git commit -m "Your descriptive message"
       git push origin your-feature-branch

Multiple Terminal Sessions
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can open multiple terminals into the same container:

.. code-block:: bash

    # Terminal 1 (host)
    docker exec -it tidy3d_python_client_dev bash

    # Terminal 2 (host)
    docker exec -it tidy3d_python_client_dev bash

    # Terminal 3 (host)
    docker exec -it tidy3d_python_client_dev bash

Each terminal shares the same filesystem and running processes.

Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

    # Inside container with venv activated

    # Run all tests
    pytest

    # Run tests in parallel (faster)
    pytest -n auto

    # Run specific test module
    pytest tests/test_components.py

    # Run tests matching pattern
    pytest -k "test_simulation"

    # Run with coverage
    pytest --cov=tidy3d --cov-report=html
    # View coverage: python -m http.server 8000 -d htmlcov

Code Quality Checks
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Inside container with venv activated

    # Format code with ruff
    ruff format .

    # Check for linting issues
    ruff check .

    # Auto-fix linting issues
    ruff check --fix .

    # Run type checking
    mypy tidy3d

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Inside container with venv activated
    cd docs

    # Build HTML documentation
    make clean
    make html

    # View the docs
    cd _build/html
    python -m http.server 8000
    # Open http://localhost:8000 in your host browser

Managing the Container
----------------------

Start and Stop
^^^^^^^^^^^^^^

.. code-block:: bash

    # Start container
    docker start tidy3d_python_client_dev

    # Stop container (preserves state)
    docker stop tidy3d_python_client_dev

    # Restart container
    docker restart tidy3d_python_client_dev

Checking Status
^^^^^^^^^^^^^^^

.. code-block:: bash

    # Check if container is running
    docker ps

    # Check all containers (including stopped)
    docker ps -a

    # View container logs
    docker logs tidy3d_python_client_dev

    # Follow container logs (live)
    docker logs -f tidy3d_python_client_dev

Removing and Recreating
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Stop and remove container
    docker stop tidy3d_python_client_dev
    docker container rm tidy3d_python_client_dev

    # Recreate container (preserves code on host)
    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -p 8888:8888 \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

.. note::
    Your code changes are safe! They're stored on your host in the mounted directory.

Rebuilding the Image
^^^^^^^^^^^^^^^^^^^^

If ``dev.Dockerfile`` changes or you want to update base dependencies:

.. code-block:: bash

    # Rebuild the image
    docker build -t tidy3d-python-client-dev -f dev.Dockerfile .

    # Remove old container and create new one
    docker stop tidy3d_python_client_dev
    docker container rm tidy3d_python_client_dev
    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -p 8888:8888 \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

Advanced Usage
--------------

Running Jupyter Lab
^^^^^^^^^^^^^^^^^^^

Start Jupyter Lab inside the container to work with notebooks:

.. code-block:: bash

    # Inside container with venv activated
    cd tidy3d
    jupyter lab . --ip=0.0.0.0 --port=8888

Jupyter Lab will print a URL like:

.. code-block::

    http://127.0.0.1:8888/lab?token=abc123def456...

Copy this complete URL and paste it into your web browser (like Firefox) on your host machine to begin your session.

.. note::
    Port 8888 is forwarded to your host via the ``-p 8888:8888`` flag used when creating the container.

Running GitHub Actions Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run GitHub Actions locally, install the `act <https://github.com/nektos/gh-act>`_ extension:

.. code-block:: bash

    gh extension install nektos/gh-act

This repository no longer tracks GitHub Actions workflow files, so there is nothing here to run with ``gh act``.

VSCode Integration
^^^^^^^^^^^^^^^^^^

Attach VSCode to the running container for a full IDE experience:

**Install Extensions**

In VSCode, install:

* **Remote - Containers** (ms-vscode-remote.remote-containers)
* **Docker** (ms-azuretools.vscode-docker)

**Attach to Container**

1. Start your container:

   .. code-block:: bash

       docker start tidy3d_python_client_dev

2. In VSCode:

   * Open Command Palette (Cmd/Ctrl+Shift+P)
   * Type "Remote-Containers: Attach to Running Container"
   * Select ``tidy3d_python_client_dev``

3. VSCode will open a new window connected to the container

4. Open folder: ``/home/flexdaemon/tidy3d``

5. Select Python interpreter: ``.venv/bin/python``

Now you can:

* Edit code with full IntelliSense
* Run tests from VSCode test explorer
* Debug Python code with breakpoints
* Use integrated terminal (already inside container)

Optional: AWS CodeArtifact Access (Internal Developers Only)
-------------------------------------------------------------

.. note::
    **Who needs this?**

    CodeArtifact hosts internal packages like ``tidy3d-extras`` needed for:

    * Integration tests in CI/CD
    * Backend-dependent development work
    * Full-stack Tidy3D development

    **External contributors**: You can skip this entire section. The open-source ``tidy3d`` client works without CodeArtifact.

.. warning::
    **Security Policy**: We use AWS SSO for authentication, which requires daily login for security compliance.
    This is a simple 2-step process per development session:

    1. ``aws sso login`` - Opens browser for authentication (~30 seconds)
    2. Export ``UV_INDEX_CODEARTIFACT_*`` credentials from AWS CodeArtifact

Why CodeArtifact?
^^^^^^^^^^^^^^^^^

**Q: Why can't we use PyPI?**

A: Internal packages like ``tidy3d-extras`` contain proprietary code that cannot be published publicly.

**Q: Why SSO with daily expiration?**

A: Flexcompute security policy requires time-limited credentials. Long-lived tokens pose security risks.

**Q: What if I don't want to deal with this?**

A: Use the Docker development environment, which matches our CI setup exactly. Or if you're only working on the open-source client, you don't need CodeArtifact at all.

Supported Path: uv with AWS CLI Credentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The development container includes AWS CLI v2, which is enough to authenticate uv against CodeArtifact.

.. note::
    **For non-Docker users**: Ensure AWS CLI v2 is installed on your system:

    .. code-block:: bash

        aws --version  # Should show version 2.x

**One-Time Setup**

Inside the container (or on your local machine if not using Docker), configure AWS SSO (only needed once):

.. code-block:: bash

    # Configure AWS SSO
    aws configure sso

Enter the following when prompted:

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Prompt
      - Value
    * - SSO session name
      - ``flexcompute-pypi``
    * - SSO start URL
      - ``https://d-9067bfae6e.awsapps.com/start/#``
    * - SSO region
      - ``us-east-1``
    * - Account
      - ``625554095313``
    * - Role
      - ``codeartifact-readonly``
    * - Profile name
      - ``flexcompute-pypi``

**Daily Authentication** (each development session)

Each day when you start development, run these two commands:

.. code-block:: bash

    # Step 1: Login to AWS SSO (opens browser on host for authentication)
    aws sso login --profile flexcompute-pypi

    # Step 2: Export CodeArtifact credentials for uv
    export UV_INDEX_CODEARTIFACT_USERNAME=aws
    export UV_INDEX_CODEARTIFACT_PASSWORD="$(aws codeartifact get-authorization-token \
        --domain flexcompute \
        --domain-owner 625554095313 \
        --region us-east-1 \
        --profile flexcompute-pypi \
        --query authorizationToken \
        --output text)"

    # Install project dependencies (including the extras plugin)
    uv sync --frozen --extra dev --extra extras

.. note::
    **How it works**: the exported uv credentials:

    * Generates a short-lived CodeArtifact authentication token
    * Configure uv to use that token for the ``codeartifact`` source
    * Expire automatically after the AWS CodeArtifact token lifetime

Persisting AWS Credentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid re-configuring AWS SSO each time you recreate the container, mount your AWS credentials:

.. code-block:: bash

    # Recreate container with AWS credentials mounted
    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -p 8888:8888 \
        -v .:/home/flexdaemon/tidy3d \
        -v ~/.aws:/home/flexdaemon/.aws \
        tidy3d-python-client-dev

This way, you only need to run ``aws configure sso`` once on your host machine.

Alternative Package Managers (Self-Support)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer ``pip`` or another package manager, you'll need to configure
CodeArtifact authentication yourself.

**The officially supported development path uses uv.** For other package managers:

* See `AWS CodeArtifact pip documentation <https://docs.aws.amazon.com/codeartifact/latest/ug/python-configure-pip.html>`_
* Use ``aws codeartifact login --tool pip`` for manual setup
* You are responsible for configuration and troubleshooting

Alternatively, use the Docker development environment which has everything pre-configured.

Quick Reference
---------------

Command Cheatsheet
^^^^^^^^^^^^^^^^^^

**Container Management:**

.. code-block:: bash

    # Build image
    docker build -t tidy3d-python-client-dev -f dev.Dockerfile .

    # Create container
    docker container create --name=tidy3d_python_client_dev \
        --userns=host -p 8888:8888 \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

    # Start/stop container
    docker start tidy3d_python_client_dev
    docker stop tidy3d_python_client_dev

    # Enter container
    docker exec -it tidy3d_python_client_dev bash

**Inside Container:**

.. code-block:: bash

    # Setup environment
    cd tidy3d
    uv venv -p 3.11
    source .venv/bin/activate
    uv pip install -e .[dev]

    # Testing
    pytest                          # All tests
    pytest -n auto                  # Parallel tests
    pytest tests/test_file.py       # Specific file

    # Code quality
    ruff format .                   # Format code
    ruff check .                    # Lint code
    mypy tidy3d                     # Type check

    # Documentation
    cd docs && make html            # Build docs

    # Jupyter
    jupyter lab . --ip=0.0.0.0      # Start Jupyter Lab

    # CodeArtifact (internal developers only)
    aws sso login --profile flexcompute-pypi                    # Daily: Login to AWS
    export UV_INDEX_CODEARTIFACT_USERNAME=aws
    export UV_INDEX_CODEARTIFACT_PASSWORD="$(aws codeartifact get-authorization-token --domain flexcompute --domain-owner 625554095313 --region us-east-1 --profile flexcompute-pypi --query authorizationToken --output text)"

Directory Structure
^^^^^^^^^^^^^^^^^^^

.. code-block::

    Host:
    ~/flexcompute/tidy3d/
    ├── tidy3d/              # Source code (mounted to container)
    ├── tests/               # Test suite (mounted)
    ├── docs/                # Documentation (mounted)
    ├── .venv/               # Virtual env (created inside, persists)
    ├── pyproject.toml       # uv config (mounted)
    └── dev.Dockerfile       # Docker image definition

    Container:
    /home/flexdaemon/
    ├── tidy3d/              # Mounted from host
    │   ├── .venv/           # Your virtual environment
    │   ├── tidy3d/          # Package source
    │   └── tests/           # Tests
    └── .aws/                # AWS credentials (if mounted)

Troubleshooting
---------------

Container Issues
^^^^^^^^^^^^^^^^

**Problem**: ``docker: permission denied``

**Solution**: Add your user to the docker group:

.. code-block:: bash

    sudo usermod -aG docker $USER
    # Logout and login (or reboot)

**Problem**: Container exits immediately

**Solution**: Check logs for errors:

.. code-block:: bash

    docker logs tidy3d_python_client_dev

    # Try running interactively to debug
    docker run --rm -it tidy3d-python-client-dev bash

**Problem**: Port 8888 already in use

**Solution**: Use a different port:

.. code-block:: bash

    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -p 8889:8888 \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

    # Access Jupyter at http://localhost:8889

Permission Issues
^^^^^^^^^^^^^^^^^

**Problem**: Cannot edit files, permission denied

**Solution**: Ensure container uses host user namespace:

.. code-block:: bash

    # Container must be created with --userns=host flag
    docker container create --name=tidy3d_python_client_dev \
        --userns=host \
        -v .:/home/flexdaemon/tidy3d \
        tidy3d-python-client-dev

**Problem**: ``flexdaemon`` user has wrong UID

**Solution**: The ``dev.Dockerfile`` creates user with UID 1000. If your host user has a different UID, files may have permission issues. Check your host UID:

.. code-block:: bash

    # On host
    id -u  # Should output 1000 for best compatibility

    # If different, you can modify the Dockerfile or change file ownership

Virtual Environment Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: ``uv venv`` command not found

**Solution**: ``uv`` is pre-installed in the image. Verify:

.. code-block:: bash

    which uv
    uv --version

    # If missing, rebuild the image

**Problem**: Virtual environment activation fails

**Solution**: Make sure you're in the right directory:

.. code-block:: bash

    cd /home/flexdaemon/tidy3d
    ls .venv  # Should exist after 'uv venv'
    source .venv/bin/activate

Test Failures
^^^^^^^^^^^^^

**Problem**: Tests fail with ``ModuleNotFoundError``

**Solution**: Reinstall in editable mode:

.. code-block:: bash

    # Inside container with venv activated
    cd /home/flexdaemon/tidy3d
    uv pip install -e .[dev]

**Problem**: Import errors for optional dependencies

**Solution**: Install all extras:

.. code-block:: bash

    uv pip install -e .[dev,vtk,trimesh,gdstk,pytorch]

CodeArtifact Authentication Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: ``uv sync --frozen --extra dev --extra extras`` fails with authentication errors

**Solution**: Verify SSO session is active and re-authenticate:

.. code-block:: bash

    # Check if SSO session is valid
    aws sts get-caller-identity --profile flexcompute-pypi

    # If expired or fails, re-authenticate (2-step process)
    aws sso login --profile flexcompute-pypi
    export UV_INDEX_CODEARTIFACT_USERNAME=aws
    export UV_INDEX_CODEARTIFACT_PASSWORD="$(aws codeartifact get-authorization-token \
        --domain flexcompute \
        --domain-owner 625554095313 \
        --region us-east-1 \
        --profile flexcompute-pypi \
        --query authorizationToken \
        --output text)"

    # Retry installation
    uv sync --frozen --extra dev --extra extras

**Problem**: ``aws sso login`` doesn't work, times out, or fails

**Solution**: This is an AWS account access issue. Contact IT support or your team lead to verify:

* Your AWS account is active
* You have access to the correct AWS organization
* Your SSO credentials are correct
* You have the ``codeartifact-readonly`` role assigned

**Problem**: I use pip, not uv. How do I authenticate?

**Solution**: The officially supported path is uv with CodeArtifact credentials in
``UV_INDEX_CODEARTIFACT_USERNAME`` and ``UV_INDEX_CODEARTIFACT_PASSWORD``.

For pip users, you must self-support using AWS documentation.

**Option 1 (for pip users):**

.. code-block:: bash

    aws sso login --profile flexcompute-pypi
    aws codeartifact login --tool pip \
        --repository pypi-releases \
        --domain flexcompute \
        --domain-owner 625554095313 \
        --region us-east-1 \
        --profile flexcompute-pypi

**Option 2 (not recommended):**

.. code-block:: bash

    CODEARTIFACT_AUTH_TOKEN=`aws codeartifact get-authorization-token \
        --domain flexcompute \
        --domain-owner 625554095313 \
        --query authorizationToken \
        --output text \
        --profile flexcompute-pypi`

    pip config set site.extra-index-url \
        https://aws:$CODEARTIFACT_AUTH_TOKEN@flexcompute-625554095313.d.codeartifact.us-east-1.amazonaws.com/pypi/pypi-releases/simple/

Note: Tokens expire after 12 hours, requiring daily re-authentication.

Alternatively, use the Docker development environment.

**Problem**: ``uv lock`` or ``uv update`` hangs

**Solution**: Clear uv cache and retry:

.. code-block:: bash

    # Clear uv cache
    uv cache clear pypi --all
    uv cache clear codeartifact --all

    # Retry operation
    uv lock --upgrade

Jupyter Issues
^^^^^^^^^^^^^^

**Problem**: Cannot access Jupyter Lab from host browser

**Solution**: Ensure Jupyter is binding to 0.0.0.0:

.. code-block:: bash

    # Inside container
    jupyter lab . --ip=0.0.0.0 --port=8888

    # Port must be forwarded when creating container
    # Check with: docker inspect tidy3d_python_client_dev | grep PortBindings

**Problem**: Jupyter kernel dies or doesn't start

**Solution**: Check virtual environment has ipykernel:

.. code-block:: bash

    uv pip install ipykernel
    python -m ipykernel install --user --name=tidy3d

Next Steps
----------

After completing this guide, you should have:

✅ Docker development container running
✅ Python environment set up with dependencies
✅ Tests passing successfully
✅ (Optional) Jupyter Lab accessible
✅ (Optional) VSCode attached to container
✅ (Optional) AWS CodeArtifact configured

**Continue learning:**

* :doc:`usage` - Advanced development workflows
* :doc:`documentation` - Building and writing documentation
* :doc:`recommendations` - Best practices and code style
* :doc:`release/index` - Release process and versioning

**More Docker resources:**

* `Docker Documentation <https://docs.docker.com/>`_
* `Dev Containers in VSCode <https://code.visualstudio.com/docs/devcontainers/containers>`_

Welcome to Docker-based Tidy3D development!
