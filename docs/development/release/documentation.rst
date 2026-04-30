Documentation Release
---------------------

Tidy3D documentation is published by Flexcompute release automation. This public page covers source updates and local validation only; internal publishing instructions live with Flexcompute's workflow configuration.

Local docs development is unchanged:

.. code-block:: bash

    uv run tidy3d develop build-docs

That command builds the HTML into ``_docs/`` for local inspection.

Hot Fix & Submodule Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make a documentation hot fix, update the relevant documentation source or advance the notebook / FAQ content that should appear in the next documentation publish. Then build locally before handing the change to the normal release process.
