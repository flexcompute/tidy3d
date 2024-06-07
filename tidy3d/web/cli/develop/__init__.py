# Import from documentation.py
from .documentation import (
    build_documentation,
    # build_documentation_pdf,
    build_documentation_from_remote_notebooks,
    commit,
    # convert_all_markdown_to_rst_command,
    replace_in_files_command,
)

# Import from index.py
from .index import develop

# Import from install.py
from .install import (
    activate_correct_poetry_python,
    configure_submodules,
    get_install_directory_command,
    install_development_environment,
    install_in_poetry,
    uninstall_development_environment,
    update_submodules_remote,
    verify_development_environment,
    verify_pandoc_is_installed_and_version_less_than_3,
    verify_pipx_is_installed,
    verify_poetry_is_installed,
    verify_sphinx_is_installed,
)
from .packaging import benchmark_timing_operations, benchmark_timing_operations_command

# Import from tests.py
from .tests import test_in_environment_command, test_options

# Import from utils.py
from .utils import echo_and_check_subprocess, echo_and_run_subprocess, get_install_directory

__all__ = [
    "benchmark_timing_operations",
    "benchmark_timing_operations_command",
    "build_documentation",
    # "build_documentation_pdf",
    "build_documentation_from_remote_notebooks",
    "commit",
    # "convert_all_markdown_to_rst_command",
    "replace_in_files_command",
    "test_options",
    "test_in_environment_command",
    "activate_correct_poetry_python",
    "configure_submodules",
    "verify_pandoc_is_installed_and_version_less_than_3",
    "verify_pipx_is_installed",
    "verify_poetry_is_installed",
    "verify_sphinx_is_installed",
    "get_install_directory_command",
    "install_development_environment",
    "install_in_poetry",
    "uninstall_development_environment",
    "update_submodules_remote",
    "verify_development_environment",
    "get_install_directory",
    "echo_and_run_subprocess",
    "echo_and_check_subprocess",
    "develop",
]
