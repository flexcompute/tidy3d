Recommendations
=================

Standardised Commit Messages
----------------------------

Now, realistically, this is a `matter of preference <https://medium.com/walmartglobaltech/semantic-commit-messages-with-emojis-dba2541cea9a>`_.
However, it could be argued there is something nice in having standard commit messages that can be easily searched through
to understand the role of each change, and also render nicely in the git history. Also, having a commit standard maybe makes people
looking through our code feel that we take pride in our work and also like to make it nice. It is debatable whether this is a way to do this, however, we can update these recommendations depending on how we consider best.

However, if we do decide to commit with emojis, I believe it would be worth having a standard, so that it does not get polluted with different emojis (as I have been guilty of before) and also as can be seen in other open-source projects.

.. list-table:: Commit Standard
    :header-rows: 1
    :widths: 25 15 15 45

    * - Purpose
      - Emoji
      - Types
      - Example
    * - |:sparkles:| New Feature
      - ``:sparkles:``
      - ``FEAT:``
      - ``:sparkles: FEAT: <my commit message>``
    * - |:wrench:| Fix Broken Code
      - ``:wrench:``
      - ``FIX:``
      - ``:wrench: FIX: <my commit message>``
    * - |:package:| Packaging-related
      - ``:package:``
      - ``BUILD:``
      - ``:package: BUILD: <my commit message>``
    * - |:book:| Documentation-related
      - ``:book:``
      - ``DOCS:``
      - ``:book: DOCS: <my commit message>``
    * - |:rocket:| Refactor code
      - ``:rocket:``
      - ``REFC:``
      - ``:rocket: REFC: <my commit message>``
    * - |:test_tube:| Testing related
      - ``:test_tube:``
      - ``TEST:``
      - ``:test_tube: TEST: <my commit message>``
    * - |:tada:| Release commit
      - ``:tada:``
      - ``RELEASE:``
      - ``:tada: RELEASE: <my commit message>``


Package Speedup Best Practices
--------------------------------

``tidy3d`` is a pretty big project already, and will get bigger. We want to optimise the performance of the codebase throughout the multiple operations that we perform.

We want to improve the speed of the project ``import`` and there are a few techniques to do this which are inherent to the way we write our code.

- `For profiling the package distribution in the import <https://stackoverflow.com/questions/16373510/improving-speed-of-python-module-import>`_
- `In terms of speeding our operations <https://wearecommunity.io/communities/tectoniques/articles/2499#:~:text=Proper%20Import,in%20slowing%20down%20code%20performance.>`_

We have already begun facing these type of code-speed issues as first raised `here <https://github.com/flexcompute/tidy3d/pull/1300>`_, `here <https://github.com/flexcompute/tidy3d/pull/1300>`_

So when we import dependencies inside our code-base in particular where these are used, we will try to do the following:

.. code::

    from mypackage import just_what_I_need

instead of

.. code::

    import mypackage

This is because the latter will import the entire package, which is not necessary and will slow down the code.


Managing Optional Dependencies On-The-Fly
------------------------------------------


If you look within ``pyproject.toml``, it is possible to see that we have different packages relating to different functionalities that are optional.

Some examples from these are ``[vtk, jax, trimesh, gdstk, gdspy]`` etc. What we want to do is improve the import speed of the core-package in order to minimise small core operations. As we scale into a bigger package, decoupling these type of imports from the total pacakge import is essential.


Benchmarking Package Import
----------------------------

We want to make the tidy3d package be as light as possible for a given set of operations. As such, it is important to understand exactly where a given set of operations is expending computational power.

We have a set of utilties to verify this.

.. code::

    poetry run tidy3d develop benchmark-timing-operations