#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
NOTEBOOKS_PATH="docs/notebooks"
FAQ_PATH="docs/faq"

# Initialize and update submodules
echo "Initializing and updating submodules..."
git submodule update --init --recursive

# Function to check if a submodule is up to date
check_submodule() {
  local path=$1
  local branch=$2
  echo "Checking $path for updates..."
  cd $path
  local current_commit=$(git rev-parse HEAD)
  echo $(git fetch --all --verbose)
  echo $(git remote get-url origin)

  if git show-ref --verify refs/remotes/origin/$branch; then
    echo "Branch $branch exists."
  else
    echo "::error::Branch $branch does not exist on remote."
    exit 1
  fi

  local latest_commit=$(git rev-parse refs/remotes/origin/$branch)
  echo "LATEST_COMMIT: $latest_commit"
  echo "CURRENT_COMMIT: $current_commit"

  cd - > /dev/null
  if [ "$latest_commit" != "$current_commit" ]; then
    echo "::error::Submodule $path is not up to date with the $branch branch. Please update it."
    exit 1
  else
    echo "Submodule $path is up to date with the $branch branch."
  fi
}

# Check Notebooks submodule with the same branch as the main project
check_submodule $NOTEBOOKS_PATH $BRANCH_NAME

# Check FAQs submodule only on the develop branch
if [ "$BRANCH_NAME" == "develop" ]; then
  check_submodule $FAQ_PATH "develop"
fi

echo "All submodules are up to date."
