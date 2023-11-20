Development Flow
=========================

Utilities
^^^^^^^^^^

There are a range of handy development functions that you might want to use to streamline your development experience.

.. list-table:: Use Cases
   :header-rows: 1

    * - Description
      - Caveats
      - Command
    * - Dual snapshot between the ``tidy3d`` and ``notebooks`` source and submodule repository.
      - Make sure you are on the correct git branches you wish to commit to on both repositories, and all `non-git-ignored` files will be added to the commit.
      - ``tidy3d develop commit <your message>``
