#!/bin/bash

# Function to verify if Pandoc is installed and its version is less than 3
verify_pandoc() {
    pandoc_version=$(pandoc --version 2>&1 | grep "pandoc" | awk '{print $2}')
    if [[ $? -ne 0 ]]; then
        echo "Pandoc is not installed or not found in the system PATH."
        return 1
    fi

    major_version=$(echo "$pandoc_version" | cut -d. -f1)
    if [[ $major_version -lt 3 ]]; then
        echo "Pandoc is installed with version $pandoc_version, which is less than 3."
    else
        echo "Pandoc version $pandoc_version is installed, but it is not less than 3."
        return 1
    fi
}

# Function to verify if pipx is installed
verify_pipx() {
    python3 -m pipx --version &> /dev/null
    if [[ $? -eq 0 ]]; then
        echo "pipx is installed."
    else
        echo "pipx is not installed or not found in the system PATH."
        return 1
    fi
}

# Function to verify if Poetry is installed
verify_poetry() {
    poetry --version &> /dev/null
    if [[ $? -eq 0 ]]; then
        echo "Poetry is installed."
    else
        echo "Poetry is not installed or not found in the system PATH."
        return 1
    fi
}

# Function to verify the development environment
verify_development_environment() {
    verify_pipx
    verify_poetry
    verify_pandoc
    poetry env use python3
    poetry install -E dev --dry-run
    echo "`poetry install -E dev` dry run on the `poetry.lock` complete."
}

# Function to configure the development environment
configure_development_environment() {
    # Verify and install pipx if required
    if ! verify_pipx; then
        echo "Installing pipx..."
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
    fi

    # Verify and install poetry if required
    if ! verify_poetry; then
        echo "Installing Poetry..."
        python3 -m pipx install poetry
    fi

    # Verify pandoc is installed and version is less than 3
    if ! verify_pandoc; then
        echo "Please install Pandoc < 3 from https://pandoc.org/installing.html and rerun this script."
    fi

    # Install development dependencies
    echo "Installing development dependencies with Poetry..."
    poetry install -E dev

    # Configure notebook submodule
    # Assuming git is installed and the script is run from the repository root
    echo "Configuring notebook submodule..."
    git submodule init
    git submodule update --remote
    echo "Notebook submodule updated from remote."
}


# Function to commit changes in a Git repository and its submodule
commit_changes() {
    local message=$1
    local submodule_path=${2:-"./docs/notebooks"}

    git -C "$submodule_path" add .
    git -C "$submodule_path" commit --no-verify -am "$message"

    git add .
    git commit --no-verify -am "$message"
}

# Function to build documentation
build_documentation() {
    poetry run python -m sphinx docs/ build_docs/
}

# Command line argument parsing (can be expanded as needed)
case "$1" in
    verify-dev-environment)
        verify_development_environment
        ;;
    configure-dev-environment)
        configure_development_environment
        ;;
    commit)
        commit_changes "$2" "$3"
        ;;
    build-docs)
        build_documentation
        ;;
    *)
        echo "Usage: $0 {verify-dev-environment|configure-dev-environment|commit|build-docs}"
        ;;
esac
