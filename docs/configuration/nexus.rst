.. currentmodule:: tidy3d

Nexus Environment Configuration
================================

Overview
--------

The Nexus environment enables Tidy3D to connect to custom on-premises or private cloud deployments for enterprise customers.

.. note::
   This feature is for enterprise customers with custom Tidy3D deployments. Most users should use the standard API key setup with the default production environment. See the :doc:`../install` guide for standard setup instructions.

Quick Start
-----------

**Option 1: Automatic Configuration** (Recommended)

When you configure Nexus using the CLI, Tidy3D automatically sets it as your default profile:

.. code-block:: bash

   tidy3d configure --apikey YOUR_KEY \
     --nexus-url http://your-server

This automatically:

- Sets API endpoint: ``http://your-server/tidy3d-api``
- Sets Website endpoint: ``http://your-server/tidy3d``
- Sets S3 endpoint: ``http://your-server:9000``
- **Sets Nexus as the default profile** (persisted across sessions)

After configuration, all your scripts will automatically use Nexus without needing ``config.switch_profile("nexus")``.

**Option 2: Manual Profile Switching** (for temporary use):

For local development on localhost without persistent configuration:

.. code-block:: python

   from tidy3d.config import config
   config.switch_profile("nexus")

This uses the built-in nexus profile with localhost endpoints (127.0.0.1:5000 for API, 127.0.0.1:9000 for S3).

**Advanced setup** with individual endpoints:

.. code-block:: bash

   tidy3d configure --apikey YOUR_KEY \
     --api-endpoint http://your-server/tidy3d-api \
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
* ``--nexus-url <url>``: Nexus base URL (automatically sets api=/tidy3d-api, web=/tidy3d, s3=:9000)
* ``--api-endpoint <url>``: Nexus API server URL (e.g., http://server/tidy3d-api)
* ``--website-endpoint <url>``: Nexus web interface URL (e.g., http://server/tidy3d)
* ``--s3-region <region>``: S3 region (default: us-east-1)
* ``--s3-endpoint <url>``: S3 storage URL (e.g., http://server:9000)
* ``--ssl-verify`` / ``--no-ssl-verify``: SSL verification (default: enabled for HTTPS)
* ``--enable-caching`` / ``--no-caching``: Server-side result caching

Examples
~~~~~~~~

.. code-block:: bash

   # Simple configuration using nexus-url (recommended)
   tidy3d configure --apikey YOUR_KEY \
     --nexus-url http://nexus.company.com

   # Configure individual endpoints
   tidy3d configure --apikey YOUR_KEY \
     --api-endpoint http://nexus.company.com/tidy3d-api \
     --website-endpoint http://nexus.company.com/tidy3d

   # Add Nexus to existing configuration (preserves API key)
   tidy3d configure --nexus-url http://nexus.company.com

   # Full configuration with all options
   tidy3d configure --apikey YOUR_KEY \
     --api-endpoint https://api.company.com/tidy3d-api \
     --website-endpoint https://tidy3d.company.com \
     --s3-region eu-west-1 \
     --s3-endpoint https://s3.company.com:9000 \
     --ssl-verify \
     --enable-caching

Configuration File
~~~~~~~~~~~~~~~~~~

**Location:**

When you configure Nexus, your custom settings are saved to a profile file:

- Linux/macOS: ``~/.config/tidy3d/profiles/nexus.toml``
- Windows: ``C:\Users\username\.config\tidy3d\profiles\nexus.toml``

The base configuration file (``~/.config/tidy3d/config.toml``) stores the API key and default profile setting:

.. code-block:: toml

   # Base config: ~/.config/tidy3d/config.toml
   config_version = 1
   default_profile = "nexus"  # Profile to use by default

   [web]
   apikey = "your-api-key"

**Nexus Profile Format** (``profiles/nexus.toml``):

.. code-block:: toml

   # Nexus profile: ~/.config/tidy3d/profiles/nexus.toml
   config_version = 1
   [web]
   api_endpoint = "http://nexus.company.com/tidy3d-api"
   website_endpoint = "http://nexus.company.com/tidy3d"
   s3_region = "us-east-1"
   ssl_verify = false
   enable_caching = false

   [web.env_vars]
   AWS_ENDPOINT_URL_S3 = "http://nexus.company.com:9000"

The nexus profile file contains only your custom nexus-specific settings (endpoints, SSL, caching). The API key is stored in the base config and shared across all profiles. When the nexus profile is loaded, these settings override the built-in nexus defaults (localhost:5000). The ``default_profile`` setting in the base config tells Tidy3D to automatically load the nexus profile on startup.

Python Usage
------------

After CLI Configuration (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've configured Nexus using the CLI, **no code changes are required**. Tidy3D automatically uses your Nexus endpoints:

.. code-block:: python

   import tidy3d as td
   import tidy3d.web as web

   # After running: tidy3d configure --nexus-url http://your-server
   # Your scripts automatically use Nexus - no profile switching needed!

   sim = td.Simulation(...)
   sim_data = web.run(sim, task_name="my_sim")

The CLI configuration persists across sessions, so you only need to configure once.

Manual Profile Switching (Temporary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For temporary use or local development without persistent configuration:

.. code-block:: python

   from tidy3d.config import config

   # Temporarily switch to the built-in nexus profile (localhost defaults)
   config.switch_profile("nexus")

   # Now run simulations against your local Nexus instance
   import tidy3d as td
   import tidy3d.web as web

   sim = td.Simulation(...)
   sim_data = web.run(sim, task_name="my_sim")

The built-in ``nexus`` profile uses these default localhost settings:

- API endpoint: ``http://127.0.0.1:5000``
- Website endpoint: ``http://127.0.0.1/tidy3d``
- S3 endpoint: ``http://127.0.0.1:9000``
- SSL verification: disabled
- Server-side caching: disabled

Switching Back to Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Using CLI** (Recommended)

.. code-block:: bash

   tidy3d configure --restore-defaults

This will:
- Clear the default_profile setting
- Remove the nexus profile file
- Revert to production endpoints

**Option 2: Using Python**

.. code-block:: python

   from tidy3d.config import config

   # Clear the default profile (reverts to production)
   config.set_default_profile(None)
   config.save()

**Option 3: Switch profiles per session**

.. code-block:: python

   from tidy3d.config import config

   # Temporarily switch to production
   config.switch_profile("prod")

Verifying Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Check your current Nexus configuration:

.. code-block:: python

   from tidy3d import config

   # Check which profile is active and which is default
   print(f"Active profile: {config.profile}")
   print(f"Default profile: {config.get_default_profile()}")

   # Display current configuration
   print(config.format())

   # Check specific web settings
   print(f"API Endpoint: {config.web.api_endpoint}")
   print(f"Website: {config.web.website_endpoint}")
   print(f"S3 Region: {config.web.s3_region}")
   print(f"SSL Verify: {config.web.ssl_verify}")

   # Check S3 endpoint (stored in env_vars)
   if "AWS_ENDPOINT_URL_S3" in config.web.env_vars:
       print(f"S3 Endpoint: {config.web.env_vars['AWS_ENDPOINT_URL_S3']}")

Expected output after Nexus configuration:

.. code-block:: text

   Active profile: nexus
   Default profile: nexus
   ...

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update Nexus settings from Python:

.. code-block:: python

   from tidy3d import config
   
   # Update Nexus endpoints
   config.update_section("web",
       api_endpoint="http://nexus.company.com/tidy3d-api",
       website_endpoint="http://nexus.company.com/tidy3d",
       s3_region="us-west-2",
       ssl_verify=False,
       enable_caching=True
   )
   
   # Set S3 endpoint
   config.update_section("web",
       env_vars={"AWS_ENDPOINT_URL_S3": "http://nexus.company.com:9000"}
   )
   
   # Save changes to disk
   config.save()

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Override any setting temporarily using environment variables:

.. code-block:: bash

   # Override API endpoint
   export TIDY3D_WEB__API_ENDPOINT="http://other-server/tidy3d-api"
   
   # Override S3 endpoint
   export TIDY3D_WEB__ENV_VARS__AWS_ENDPOINT_URL_S3="http://other-s3:9000"
   
   # Override SSL verification
   export TIDY3D_WEB__SSL_VERIFY=false

Managing Configuration
----------------------

Change Default Profile
~~~~~~~~~~~~~~~~~~~~~~~

Set or clear the default profile:

.. code-block:: python

   from tidy3d import config

   # Set nexus as default
   config.set_default_profile("nexus")

   # Clear default profile (revert to production)
   config.set_default_profile(None)

   # Check current default
   print(config.get_default_profile())

Reset to Defaults
~~~~~~~~~~~~~~~~~

Reset configuration to default values (including clearing default profile):

.. code-block:: bash

   tidy3d config reset

Or from Python:

.. code-block:: python

   from tidy3d import config
   config.reset_to_defaults()
   config.save()

View Configuration
~~~~~~~~~~~~~~~~~~

Display current configuration:

.. code-block:: bash

   python -c "from tidy3d import config; print(config.format())"

Or interactively:

.. code-block:: python

   from tidy3d import config
   
   # Print formatted configuration
   print(config)
   
   # Or just web section
   print(config.web)

Remove Configuration
~~~~~~~~~~~~~~~~~~~~

To completely remove Nexus configuration:

.. code-block:: bash

   # Linux/macOS
   rm ~/.config/tidy3d/config.toml
   
   # Or reset to defaults
   tidy3d config reset
   
   # Then reconfigure with standard API key
   tidy3d configure --apikey YOUR_API_KEY

Troubleshooting
---------------

Verify Configuration
~~~~~~~~~~~~~~~~~~~~

Check that your Nexus configuration is active:

.. code-block:: python

   from tidy3d import config
   
   print("=== Tidy3D Configuration ===")
   print(f"API Endpoint: {config.web.api_endpoint}")
   print(f"Website: {config.web.website_endpoint}")
   
   # Verify it's not using production
   if "simulation.cloud" in str(config.web.api_endpoint):
       print("??  Using production, not Nexus!")
   else:
       print("? Using custom Nexus deployment")

Test Connectivity
~~~~~~~~~~~~~~~~~

Verify your Nexus server is accessible:

.. code-block:: bash

   # Test API endpoint
   curl http://your-nexus-server/tidy3d-api/health
   
   # Or with Python
   python -c "import requests; from tidy3d import config; \
   print(requests.get(f'{config.web.api_endpoint}/health').text)"

Check S3 Configuration
~~~~~~~~~~~~~~~~~~~~~~

Verify S3 endpoint configuration:

.. code-block:: python

   from tidy3d import config
   
   if config.web.env_vars and "AWS_ENDPOINT_URL_S3" in config.web.env_vars:
       s3_endpoint = config.web.env_vars["AWS_ENDPOINT_URL_S3"]
       print(f"S3 Endpoint: {s3_endpoint}")
   else:
       print("Using default AWS S3")

Common Issues
~~~~~~~~~~~~~

**Issue: Still connecting to production instead of Nexus**

Solution: Check if the default profile is set correctly:

.. code-block:: python

   from tidy3d import config
   print(f"Active profile: {config.profile}")
   print(f"Default profile: {config.get_default_profile()}")
   print(f"API Endpoint: {config.web.api_endpoint}")

If the default profile is not ``"nexus"``, reconfigure:

.. code-block:: bash

   tidy3d configure --nexus-url http://your-server

**Issue: Configuration not taking effect**

Solution: Environment variables override the default profile. Check for overrides:

.. code-block:: bash

   # Check for environment variable overrides
   env | grep TIDY3D

If ``TIDY3D_CONFIG_PROFILE``, ``TIDY3D_PROFILE``, or ``TIDY3D_ENV`` is set, it will override the default profile from config.

**Issue: API key validation fails**

Ensure you're validating against your Nexus endpoint:

.. code-block:: bash

   tidy3d configure --apikey YOUR_KEY \
     --nexus-url http://your-nexus-server

Advanced Topics
---------------

Profile Selection Priority
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tidy3D uses the following priority order when determining which profile to load:

1. **Explicit parameter**: ``ConfigManager(profile="dev")``
2. **Environment variables**: ``TIDY3D_CONFIG_PROFILE``, ``TIDY3D_PROFILE``, or ``TIDY3D_ENV``
3. **Default profile in config**: ``default_profile`` in ``config.toml``
4. **Fallback**: ``"default"`` profile (production)

This means environment variables always override the default profile setting, allowing temporary overrides without changing the configuration file.

Multiple Configurations (Profiles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Switch between different Nexus deployments:

.. code-block:: python

   from tidy3d import config
   
   # Switch to a different profile
   config.switch_profile("dev")
   
   # Configure for dev environment
   config.update_section("web",
       api_endpoint="http://dev-nexus/tidy3d-api",
       website_endpoint="http://dev-nexus/tidy3d"
   )
   config.save()
   
   # Switch back to default
   config.switch_profile("default")

Or via environment variable:

.. code-block:: bash

   # Use dev profile
   export TIDY3D_CONFIG_PROFILE=dev
   python your_script.py

Custom S3 Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

For Nexus deployments with MinIO or other S3-compatible storage:

.. code-block:: python

   from tidy3d import config
   
   config.update_section("web",
       s3_region="us-east-1",
       env_vars={
           "AWS_ENDPOINT_URL_S3": "http://minio.company.com:9000",
           # Optional: Add other AWS config
           "AWS_ACCESS_KEY_ID": "your-key",
           "AWS_SECRET_ACCESS_KEY": "your-secret"
       }
   )
   config.save()

See Also
--------

* :doc:`/api/submit_simulations` - Submitting and managing simulations
* :doc:`../install` - Installation and API key setup
