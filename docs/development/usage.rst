Using the Development Flow
==========================

Developing ``tidy3d`` with ``poetry``
-------------------------------------

``poetry`` is an incredibly powerful tool for reproducible package development envrionments and dependency management.

If you are developing ``tidy3d``, we recommend you work within the configured ``poetry`` environment defined by ``poetry.lock``. The way to install this envrionment is simple:

.. code::

    cd tidy3d/
    poetry install -E dev

This function will install the package with all the development dependencies automatically. This means you should be able to run any functionality that is possible with ``tidy3d`` reprodicibly.

It is important to note the function above is equivalent to ``pip install tidy3d[dev]``, but by using ``poetry`` there is a guarantee of using the reproducible locked environment.


Interacting with external virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to use ``poetry`` for package development. However, there are some cases where you might need to use an external virtual environment for some operations. There are a few workarounds where you can leverage the reproducibility of the ``poetry`` managed environment with the freedom of a standard virtual envrionment. See the following example:


Common Utilities
-----------------

There are a range of handy development functions that you might want to use to streamline your development experience.

.. list-table:: Use Cases
   :header-rows: 1

    * - Description
      - Caveats
      - Command
    * - Dual snapshot between the ``tidy3d`` and ``notebooks`` source and submodule repository.
      - Make sure you are on the correct git branches you wish to commit to on both repositories, and all `non-git-ignored` files will be added to the commit.
      - ``tidy3d develop commit <your message>``