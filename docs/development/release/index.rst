Release Flow
==============

Currently most of our release development flow is made under the latest ``pre/*`` branch under the main frontend tidy3d repository. You want to fork from this latest branch to develop your feature in order for it to be included under that release.

.. You just need to make sure that the ``<same>`` branches of both ``tidy3d/`` and ``tidy3d-notebooks/`` repositories within the ``./`` and ``./docs/notebooks/`` directories are updated. The ``readthedocs`` documentation will be automatically updated through the ``sync-readthedocs-repo`` Github action.

.. toctree::
    :maxdepth: 1
    :hidden:

    flow
    requirements
    notebooks

.. include:: /development/release/flow.rst
.. include:: /development/release/requirements.rst
.. include:: /development/release/notebooks.rst