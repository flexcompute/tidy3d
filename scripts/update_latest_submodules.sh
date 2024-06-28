#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

NOTEBOOKS_PATH="docs/notebooks"
FAQ_PATH="docs/faq"

# Initialize and update submodules
echo "Initializing and updating submodules..."
git submodule update --init --recursive

# Function to check if a submodule is up to date
check_submodule() {
  local path=$1
  echo "Checking $path for updates..."

  # Move into the submodule directory
  cd $path

  # Fetch the latest commits from the origin
  echo "Fetching latest commits for $path..."
  git fetch origin

  # Get the current branch of the submodule
  local branch=$(git symbolic-ref --short HEAD)

  # Get the latest commit on the submodule's branch from the origin
  local latest_commit=$(git rev-parse origin/$branch)

  # Get the current commit of the submodule
  local current_commit=$(git rev-parse HEAD)

  echo "LATEST_COMMIT: $latest_commit"
  echo "CURRENT_COMMIT: $current_commit"

  # Compare commits and update if necessary
  if [ "$latest_commit" != "$current_commit" ]; then
    echo "::error::Submodule $path is not up to date with the $branch branch. Attempting automatic update."
    git pull origin $branch
    echo "Submodule $path updated to the latest commit on $branch."
  else
    echo "Submodule $path is up to date with the $branch branch."
  fi

  # Return to the main project directory
  cd - > /dev/null
}

# Check each submodule
check_submodule $NOTEBOOKS_PATH
check_submodule $FAQ_PATH

echo "All submodules are up to date with origin for their current branch."
