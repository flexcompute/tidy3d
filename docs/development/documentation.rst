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
^^^^^^^^^^^^^^^^^^^^^^

This process is self-contained in ``tidy3d-notebooks``.

Make sure to add a link to the notebook in ``tidy3d-notebooks/docs/*`` directory in a relevant file.

Then you have to commit to either the ``develop`` branch or your custom one. However, the important thing to understand is that the submodule in ``docs/notebooks`` has a state that is also committed. This means that when you or any tool clones this directory, then the state and mapped branch/commit of the submodule will be the one that was committed. However, you have to be careful that when your commit gets merged the commit of the ``tidy3d-notebooks`` submodule is also pointing to the latest ``develop`` branch and not any local branch in which you have been developing. Otherwise, the documentation will be built with your local branch, and not the published branch.

This submodule commit process can be done by running ``git add docs/notebooks`` and then committing the change.

If you want to locally develop notebooks in ``tidy3d/docs/notebooks`` then just use that submodule as your main development repository and commit to your local branch. Then when you are ready to publish, just make sure to commit the submodule to the latest ``develop`` branch. You can then build the documentation locally easily using this approach before it is published.

Updating Docstrings
^^^^^^^^^^^^^^^^^^^^^^

The ``tidy3d develop`` suite includes a utility command ``replace-in-files``, which is designed to recursively find and replace strings in files within a specified directory. This functionality is particularly useful for updating docstrings across the codebase when there are changes in the API, ensuring that the documentation remains consistent with multiple version updates.
This is useful when updating the API and you want to update the docstrings to reflect the changes from multiple versions.

Example usage:

.. code::

    poetry run tidy3d develop replace-in-files -d ./ -j ./docs/versions/test_replace_in_files.json -v 0.18.0 --dry-run True


**Command Details**

- **Name:** ``replace-in-files``
- **Description:** Recursively finds and replaces strings in files based on a JSON configuration.
- **Options:**
  - ``--directory`` or ``-d``: Specifies the directory to process. Defaults to the current directory if not provided.
  - ``--json-dictionary`` or ``-j``: Path to a JSON file containing the mapping of strings to be replaced.
  - ``--selected-version`` or ``-v``: Specifies the version to select from the JSON file.
  - ``--dry-run``: Executes the command in a dry run mode without making actual changes.


The JSON file should contain a dictionary where keys are version numbers and values are dictionaries of strings to find and their replacements.

Example JSON structure:

.. code-block:: json

    {
      "0.18.0": {
        "tidy3d.someuniquestringa": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringb": "tidy3d.someuniquestring2",
        "tidy3d.someuniquestringc": "tidy3d.someuniquestring2"
      }
    }


The command can be executed using the ``poetry run`` command. It requires specifying the directory, JSON dictionary, and the selected version. The ``--dry-run`` option allows you to preview changes without applying them.

**Example Command**

.. code::

    poetry run tidy3d develop replace-in-files -d ./ -j ./docs/versions/test_replace_in_files.json -v 0.18.0 --dry-run True

This example will process files in the current directory (``./``), using the replacement rules specified in ``test_replace_in_files.json`` for version ``0.18.0``. The ``--dry-run`` flag set to ``True`` ensures that changes are not actually applied, allowing for a safe preview of potential modifications.


Further Guidance
-----------------

- The sphinx warnings are OK as long as the build occurs, errors will cause the crash the build.
- Make sure all your internal API references start with ``tidy3d.<your_reference>``
- In notebooks, always have absolute links, otherwise the links will break when the user downloads them.


Writing Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

... raw::

    Normally, there are no heading levels assigned to certain characters as the structure is determined from the succession of headings. However, this convention is used in Python Developer’s Guide for documenting which you may follow:
    # with overline, for parts
    * with overline, for chapters
    = for sections
    - for subsections
    ^ for subsubsections
    " for paragraphs

