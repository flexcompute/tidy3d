.. currentmodule:: tidy3d

Nexus Environment Configuration
================================

Overview
--------

The Nexus environment enables Tidy3D to connect to custom on-premises or private cloud deployments for enterprise customers.

.. note::
   This feature is for enterprise customers with custom Tidy3D deployments. Most users should use the standard API key setup with the default production environment.

Quick Start
-----------

**Simple setup** using the convenience option:

.. code-block:: bash

   tidy3d configure --apikey YOUR_KEY \
     --nexus-url http://your-server

This automatically sets:

- API endpoint: ``http://your-server:5000``
- Website endpoint: ``http://your-server/tidy3d``
- S3 endpoint: ``http://your-server:9000``

**Advanced setup** with individual endpoints:

.. code-block:: bash

   tidy3d configure --apikey YOUR_KEY \
     --api-endpoint http://your-server:5000 \
     --website-endpoint http://your-server/tidy3d

Or configure only Nexus endpoints (preserves existing API key):

.. code-block:: bash

   tidy3d configure --nexus-url http://your-server

Configuration
-------------

Command Syntax
~~~~~~~~~~~~~~

.. code-block:: bash

   tidy3d configure [--apikey <key>] [--nexus-url <url> | --api-endpoint <url> --website-endpoint <url>] [OPTIONS]

**Options:**

* ``--apikey <key>``: API key (prompts if not provided and no Nexus options given)
* ``--nexus-url <url>``: Nexus base URL (convenience option that automatically sets all endpoints)
* ``--api-endpoint <url>``: Nexus API server URL (overridden by ``--nexus-url``)
* ``--website-endpoint <url>``: Nexus web interface URL (overridden by ``--nexus-url``)
* ``--s3-region <region>``: S3 region (default: us-east-1)
* ``--s3-endpoint <url>``: S3 storage URL (overridden by ``--nexus-url``)
* ``--ssl-verify`` / ``--no-ssl-verify``: SSL verification (default: disabled)
* ``--enable-caching`` / ``--no-caching``: Result caching (default: disabled)

Examples
~~~~~~~~

.. code-block:: bash

   # Simple configuration using nexus-url (recommended)
   tidy3d configure --apikey XXX \
     --nexus-url http://ec2-instance.compute.amazonaws.com

   # Configure individual endpoints
   tidy3d configure --apikey XXX \
     --api-endpoint http://api.company.com:5000 \
     --website-endpoint http://tidy3d.company.com

   # Add Nexus to existing configuration
   tidy3d configure --nexus-url http://api.company.com

   # With all options
   tidy3d configure --apikey XXX \
     --api-endpoint https://api.company.com \
     --website-endpoint https://tidy3d.company.com \
     --s3-region eu-west-1 \
     --s3-endpoint http://s3.company.com:9000 \
     --ssl-verify \
     --enable-caching

Configuration File
~~~~~~~~~~~~~~~~~~

Settings are stored in ``~/.tidy3d/config`` (Windows: ``C:\Users\username\.tidy3d\config``):

.. code-block:: toml

   apikey = "your-api-key"
   web_api_endpoint = "http://your-server:5000"
   website_endpoint = "http://your-server/tidy3d"
   s3_region = "us-east-1"
   s3_endpoint = "http://127.0.0.1:9000"
   ssl_verify = false
   enable_caching = false

Python Usage
------------

No code changes required. Tidy3D automatically uses the configured endpoints:

.. code-block:: python

   import tidy3d as td
   import tidy3d.web as web
   
   sim = td.Simulation(...)
   sim_data = web.run(sim, task_name="my_sim")


Removing Configuration
----------------------

Delete the entire file and reconfigure:

.. code-block:: bash

   rm ~/.tidy3d/config
   tidy3d configure --apikey=YOUR_API_KEY

Troubleshooting
---------------

**Verify configuration:**

.. code-block:: python

   from tidy3d.web.core.environment import Env
   print(Env.current.name, Env.current.web_api_endpoint)

**Test connectivity:**

.. code-block:: bash

   curl http://your-api-endpoint:5000/health
   export TIDY3D_ENV=prod && python -c "import tidy3d.web as web; web.test()"

See Also
--------

* :doc:`submit_simulations` - Submitting and managing simulations
* :doc:`../install` - Installation and API key setup
