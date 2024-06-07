Feature Contribution
-----------------------



Feature Development Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1. Create a branch off of ``develop``
""""""""""""""""""""""""""""""""""""""""

Our ``pre/x.x`` branches are where new features, bug fixes, and other things get added before a major release. To switch to the ``pre/x.x`` branch:

.. code-block:: bash

   git checkout pre/x.x

And then you can create a new branch from here with your GitHub username pre-pended:

.. code-block:: bash

   git checkout -b myusername/cool-feature

Currently most of our release development flow is made under the latest ``pre/*`` branch under the main frontend
tidy3d repository. You want to fork from this latest branch to develop your feature in order for it to be included under that release.

We are using a variation of the `gitflow
workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`__
- so this is the first thing to familiarize yourselves with. The link provided explains it very well, but to summarize: features get added to a pre-release branch (``pre/x.x``), and once all the features for a particular release have been implemented, the pre-release branch gets merged into ``develop``. The ``latest`` branch holds the code we want most users to be using. When we wish to release a new version, we simply merge the state of ``develop`` into ``latest``, propagating all of the changes at once. We will describe this process in more detail below.

When developing new features, we ask that you create a branch off of whichever branch you aim to contribute to. This is typically either the current pre-release branch named ``pre/x.x`` or ``develop`` depending on what stage of development we are currently in. You then work on your branch, and when the feature is ready to merge in, we prefer ``git rebase`` to ``git merge``. This creates a cleaner, linear history. You can read about why we do it and what a rebase is at `this link <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_. And see Momchil’s more specific notes `here <https://www.notion.so/6a7b343ee1cf4ca28fdad0a870354eee?pvs=21>`_.

Most importantly, **all contributions
should happen through a PR from a feature branch into the develop
branch.**

The extra step that we have in our workflow is to
always ``rebase and merge`` instead of simply ``merge`` branches. This
has the advantage of avoiding a mess of crossing paths and keeps the
history clean, but it does require a little more work. As an extra
advantage, once you get the hang of rebasing it also becomes a very
useful tool to prune your commits and write more meaningful commit
messages when you’re done with the work. The main purpose of this page
is to give an example of the workflow. For more information on the difference between rebasing vs merging,
see this `article <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`__.

The first thing to do when starting a new batch of work is to start from
a clean branch on your machine.

.. code-block:: bash

    # from the main tidy3d frontend repo
   git checkout pre/x.x
   git pull origin pre/x.x
   git checkout -b my_name/new_feature

2. Writing code
""""""""""""""""""""""""""""""""""""""""

Develop your code in this new branch, committing your changes when it seems like a natural time to “save your progress”.

If you are working on a new feature, make sure you add a line in the `CHANGELOG.md <https://github.com/flexcompute/Tidy3D-client-revamp/blob/develop/CHANGELOG.md>`_ file (if it exists in that repository) to summarize your changes.


3. Create a pull request on GitHub
""""""""""""""""""""""""""""""""""""""""

First, push your changes to your branch on GitHub.

In the GitHub website, create a pull request to merge your branch into ``pre/x.x``.

Write some comments or a summary of changes in your pull request to be clear about what is being added/changed and why.

Before rebasing, you should make sure you have the latest version
of ``develop``, in case other work has been merged meanwhile.

.. code-block:: bash

   git checkout pre/x.x
   git pull origin pre/x.x
   git checkout my_name/new_feature
   git rebase -i pre/x.x

This will now open an editor that will allow you to edit the commits in
the feature branch. There is plenty of explanations of the various
things you can do.


Most probably, you just want to squash some of your commits. The first
commit cannot be squashed - later commits get squashed into previous
commits.


Once you save the file and close it, a new file will open giving you a
chance to edit the commit message of the new, squashed commit, to your
liking. Once you save that file too and close it, the rebasing should
happen.

**NB**: The rebase may not work if there were conflicts with
current ``develop``. Ideally we should avoid that by making sure that
two people are never working on the same part of the code. When it
happens, you can try to resolve the conflicts,
or ``git rebase --abort`` if you want to take a step back and think
about it.

Finally, you now need to force push your branch to ``origin``, since the
rebasing has changed its history.

.. code-block:: bash

   git push -f origin my_name/new_feature


4. Submit for review
"""""""""""""""""""""

Every PR must have the following before it can be merged:

- At least one review.
- A description in the CHANGELOG of what has been done.

Every new major feature must also pass all of the following before it can be merged:

- Frontend and backend tests by the developer (unless no code has changed on one or the other), as well as a new example notebook or a modification to an existing example notebook that utilizes the new feature. Intermediate reviews can happen, but these conditions must be met for the feature to begin to be considered for a merge.
- Ensure any known limitations are listed at the top message in the PR conversation (e.g., does the feature work with the mode solver? The auto grid? Does it work, but not as well as it should?). The feature can be merged given the limitations if we make a decision to do that, but only if an error or warning is issued whenever a user could encounter them, and after the list has been moved to another PR or an issue to keep track.
- If backend changes are present, review by one of the people well-versed with the solver (Momchil, Weiliang, Shashwat, Daniil).
- If frontend changes are present, review by any member of the team and additional approval by Momchil or Tyler.
- QA from any member of the team: playing around with the new feature and trying to find limitations. The goal is not to construct one successful example but to figure out if there is any allowed usage that may be problematic. An extra example notebook may or may not come out of this.

After this, you can notify Momchil that the branch is ready to to be
merged. In the comment you can optionally also say things like “Fixes
#34”. This will then automatically link that PR to the particular issue,
and automatically close the issue.

This can be repeated as often as needed. In the end, you may end up with
a number of commits. We don’t **enforce** a single commit per feature,
but it makes the most sense if the feature is small. If the feature is
big and contains multiple meaningful commits, that is OK. In any case,
rebasing allows you to clean everything up.

**NB**: Only do this once you feel like you are fully done with that
feature, i.e. all PR comments have been addressed, etc. This is not
critical, but is nicer to only rebase in the end so as not to muddle up
the PR discussion when you force push the new branch (see below).
