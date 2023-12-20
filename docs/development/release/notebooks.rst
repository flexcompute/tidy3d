Notebooks Development
----------------------

All notebooks are now developed under the `tidy3d-notebooks <github.com/flexcompute/tidy3d-notebooks>`_, and you can also develop this submodule under ``tidy3d/notebooks``. Note that the submodule is linked to the ``develop`` branch of ``tidy3d-notebooks``.

Say, you have done some changes onto the repository in `tidy3d-notebooks` and propagated them to the remote branch, you can run the following command:

.. code-block::

    poetry run tidy3d develop build-docs-from-remote-notebooks

This command will pull the latest changes onto your notebook submodule and build the documentation.