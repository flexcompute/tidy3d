Feature Development Workflow
------------------------------

We are using a variation of the `gitflow
workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`__
- so this is the first thing to familiarize yourselves with. The
splitting of branches into ``main``, ``develop`` and separate feature
branches is as explained there. Most importantly, **all contributions
should happen through a PR from a feature branch into the ``develop``
branch.**

The extra step that we have in our workflow is to
always ``rebase and merge`` instead of simply ``merge`` branches. This
has the advantage of avoiding a mess of crossing paths and keeps the
history clean, but it does require a little more work. As an extra
advantage, once you get the hang of rebasing it also becomes a very
useful tool to prune your commits and write more meaningful commit
messages when you’re done with the work. The main purpose of this page
is to give an example of the workflow.

   for more information on the difference between rebasing vs merging,
   see this
   `article <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`__.

The first thing to do when starting a new batch of work is to start from
a clean branch on your machine.

.. code-block:: bash

    # from the main tidy3d frontend repo
   git checkout develop
   git pull origin develop
   git checkout -b my_name/new_feature


Create your feature rebase
''''''''''''''''''''''''''''''

Before rebasing, you should make sure you have the latest version
of ``develop``, in case other work has been merged meanwhile.

::

   git checkout develop
   git pull origin develop
   git checkout my_name/new_feature
   git rebase -i develop

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

Submitting to PR
'''''''''''''''''

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

