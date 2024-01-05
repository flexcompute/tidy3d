Recommendations
=================

Package Speedup Best Practices
''''''''''''''''''''''''''''''

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
''''''''''''''''''''''''''''''''''''''''''

If you look within ``pyproject.toml``, it is possible to see that we have different packages relating to different functionalities that are optional.

Some examples from these are ``[vtk, jax, trimesh, gdstk, gdspy]`` etc. What we want to do is improve the import speed of the core-package in order to minimise small core operations. As we scale into a bigger package, decoupling these type of imports from the total pacakge import is essential.

