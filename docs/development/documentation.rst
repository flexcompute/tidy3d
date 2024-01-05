Documentation
==============

Getting Started
---------------

Assuming you already have ``poetry`` and the ``tidy3d develop`` commands installed (see the instructions if not), then building the documentation is easy:

.. code::

        poetry run tidy3d develop build-docs

The output of the build will be in ``_docs/`` and you can view it by opening ``_docs/index.html`` in your browser. You might just have to click the ``index.html`` file to open it in your browser within a File Explorer.

Theme
------

Under ``docs/_static/css`` we can find ``custom.css`` which the color themes custom to Flexcompute can be found.

Common Updates
--------------

Adding a new notebook
''''''''''''''''''''''

This process is self-contained in ``tidy3d-notebooks``.

Make sure to add a link to the notebook in ``tidy3d-notebooks/docs/*`` directory in a relevant file.

Then you have to commit to either the ``develop`` branch or your custom one. However, the important thing to understand is that the submodule in ``docs/notebooks`` has a state that is also committed. This means that when you or any tool clones this directory, then the state and mapped branch/commit of the submodule will be the one that was committed. However, you have to be careful that when your commit gets merged the commit of the ``tidy3d-notebooks`` submodule is also pointing to the latest ``develop`` branch and not any local branch in which you have been developing. Otherwise, the documentation will be built with your local branch, and not the published branch.

This submodule commit process can be done by running ``git add docs/notebooks`` and then committing the change.

If you want to locally develop notebooks in ``tidy3d/docs/notebooks`` then just use that submodule as your main development repository and commit to your local branch. Then when you are ready to publish, just make sure to commit the submodule to the latest ``develop`` branch. You can then build the documentation locally easily using this approach before it is published.

Updating Docstrings
'''''''''''''''''''''

``tidy3d develop`` has a utility to update docstrings in the codebase.
This is useful when updating the API and you want to update the docstrings to reflect the changes from multiple versions.

Example usage:

.. code::

    poetry run tidy3d develop replace-in-files -d ./ -j ./docs/versions/test_replace_in_files.json -v 0.18.0 --dry-run True

