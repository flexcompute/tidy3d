Feature Contribution
=====================

How to develop a feature or make an individual contribution to a Tidy3D repository on GitHub.

1. Create a branch off of ``develop``
-------------------------------------

We use the ``master`` branch for our official release, to be updated sparingly.

Our ``develop`` branches are where new features, bug fixes, and other things get added before a major release. To switch to the ``develop`` branch:

.. code-block:: bash

   git checkout develop

And then you can create a new branch from here with your GitHub username pre-pended:

.. code-block:: bash

   git checkout -b myusername/cool-feature

2. Writing code
---------------

Develop your code in this new branch, committing your changes when it seems like a natural time to “save your progress”.

For an explanation of Tidy3D and specific guidelines for development, see the `user guide <https://www.notion.so/Tidy3D-User-Guide-f196e75cb67a4b1ea7dd67c479c44fa8?pvs=21>`_ and `developer guide <https://www.notion.so/DEPRECIATED-Tidy3D-Developer-Guide-23ceee49660e42fca06484bfcaa96b5c?pvs=21>`_, respectively.

If you are working on a new feature, make sure you add a line in the `CHANGELOG.md <https://github.com/flexcompute/Tidy3D-client-revamp/blob/develop/CHANGELOG.md>`_ file (if it exists in that repository) to summarize your changes.

3. Create a pull request on GitHub
-----------------------------------

First, push your changes to your branch on GitHub.

In the GitHub website, create a pull request to merge your branch into ``develop``.

Write some comments or a summary of changes in your pull request to be clear about what is being added/changed and why.

4. Submit for review
--------------------

Every PR must have the following before it can be merged:

- At least one review.
- A description in the CHANGELOG of what has been done.

Every new major feature must also pass all of the following before it can be merged:

- Frontend and backend tests by the developer (unless no code has changed on one or the other), as well as a new example notebook or a modification to an existing example notebook that utilizes the new feature. Intermediate reviews can happen, but these conditions must be met for the feature to begin to be considered for a merge.
- Ensure any known limitations are listed at the top message in the PR conversation (e.g., does the feature work with the mode solver? The auto grid? Does it work, but not as well as it should?). The feature can be merged given the limitations if we make a decision to do that, but only if an error or warning is issued whenever a user could encounter them, and after the list has been moved to another PR or an issue to keep track.
- If backend changes are present, review by one of the people well-versed with the solver (Momchil, Weiliang, Shashwat, Daniil).
- If frontend changes are present, review by any member of the team and additional approval by Momchil or Tyler.
- QA from any member of the team: playing around with the new feature and trying to find limitations. The goal is not to construct one successful example but to figure out if there is any allowed usage that may be problematic. An extra example notebook may or may not come out of this.
