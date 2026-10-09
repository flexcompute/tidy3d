# Tidy3D
[![PyPI
Name](https://img.shields.io/badge/pypi-tidy3d-blue?style=for-the-badge)](https://pypi.python.org/pypi/tidy3d)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tidy3d.svg?style=for-the-badge)](https://pypi.python.org/pypi/tidy3d/)
[![Documentation](https://img.shields.io/badge/docs-flexcompute-00643C?style=for-the-badge)](https://docs.flexcompute.com/projects/tidy3d/en/latest/)
[![License: LGPL-2.1](https://img.shields.io/badge/license-LGPL--2.1-blue?style=for-the-badge)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-5A4FCF?style=for-the-badge)](https://github.com/astral-sh/ruff)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/daquinteroflex/4702549574741e87deaadba436218ebd/raw/tidy3d_extension.json)
[![Notebooks](https://img.shields.io/badge/Demo-Live%20notebooks-8A2BE2?style=for-the-badge)](https://www.flexcompute.com/tidy3d/learning-center/example-library/)

![](https://docs.flexcompute.com/projects/tidy3d/en/latest/_static/img/Tidy3D-logo.svg)

Tidy3D is a software package for solving extremely large electrodynamics problems using the finite-difference time-domain (FDTD) method. It can be controlled through either an [open source python package](https://github.com/flexcompute/tidy3d) or a [web-based graphical user interface](https://tidy3d.simulation.cloud).

This repository contains the python API to allow you to:


* Programmatically define FDTD simulations.
* Submit and manage simulations running on Flexcompute's servers.
* Download and postprocess the results from the simulations.

![](https://docs.flexcompute.com/projects/tidy3d/en/latest/_static/img/snippet.png)

## Installation

### Signing up for tidy3d

Note that while this front end package is open source, to run simulations on Flexcompute servers requires an account with credits.
You can sign up for an account [here](https://tidy3d.simulation.cloud/signup).
After that, you can install the front end with the instructions below, or visit [this page](https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html) in our documentation for more details.

### Quickstart Installation

To install the Tidy3D Python API locally, the following instructions should work for most users.

```
pip install --user tidy3d
tidy3d configure --apikey=XXX
```

Where `XXX` is your API key, which can be copied from your [account page](https://tidy3d.simulation.cloud/account) in the web interface.

In a hosted jupyter notebook environment (eg google  colab), it may be more convenient to install and configure via the following lines at the top of the notebook.

```
!pip install tidy3d
import tidy3d.web as web
web.configure("XXX")
```

**Advanced installation instructions for all platforms is available in the [documentation installation guides](https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html).**

### Authentication Verification

To test the authentication, you may try importing the web interface via.

```
python -c "import tidy3d.web as web; web.test()"
```

It should pass without any errors if the API key is set up correctly.

### Troubleshooting

If an installation, authentication, upload, or connection problem persists, generate a diagnostic report to share with [Tidy3D support](https://www.flexcompute.com/tidy3d/technical-support/):

```bash
tidy3d troubleshoot report
```

The command prompts for a short problem description and collects Python, package, configuration, and connectivity information. Use `--no-connection` for an offline report or `--output tidy3d-support-report.txt` to save it.

Review the report before sharing it. Narrative answers and traceback files are included verbatim, and the environment section contains local paths and installed package versions. Do not share reports created with `--private-network-details` outside your institution.

See the [troubleshooting guide](https://docs.flexcompute.com/projects/tidy3d/en/latest/troubleshoot.html) for the environment-only and connection-only commands and all available options. You can also run `tidy3d troubleshoot --help` or `tidy3d troubleshoot report --help`.

To get started, our documentation has a lot of [examples](https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html) for inspiration.

## Common Documentation References

| API Resource       | URL                                                                              |
|--------------------|----------------------------------------------------------------------------------|
| Installation Guide | https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html              |
| Documentation      | https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html                |
| Example Library    | https://www.flexcompute.com/tidy3d/learning-center/example-library/ |
| FAQ                | https://www.flexcompute.com/tidy3d/learning-center/faq/             |


## FlexAgent MCP

FlexAgent connects AI clients to Tidy3D through the Model Context Protocol (MCP). For AI coding agents, install the Tidy3D plugin from the Flexcompute plugin marketplace. The plugin provides Tidy3D guidance and MCP registration backed by the independently released `tidy3d-mcp` runtime.

**Claude Code**

In Claude Code:

```text
/plugin marketplace add flexcompute/plugin-marketplace
/plugin install tidy3d@flexcompute
```

If Claude Code is already running, reload plugins after installation:

```text
/reload-plugins
```

**Codex CLI (v0.122.0+)**

Add the Flexcompute marketplace:

```bash
codex plugin marketplace add flexcompute/plugin-marketplace
```

Then open Codex, run `/plugins`, choose the Flexcompute marketplace, and install Tidy3D.

**ChatGPT desktop app (Work or Codex)**

Clone the [Flexcompute plugin marketplace](https://github.com/flexcompute/plugin-marketplace) and open the folder in the app. In Plugins, choose **Flexcompute Plugins** and install **Tidy3D**.

Configure Tidy3D through `uvx`:

```bash
uvx tidy3d configure
```

You can also set `SIMCLOUD_APIKEY` instead of running the configure command.

For advanced manual MCP setup, use Tidy3D's integrated command. It delegates to the same runtime shipped by `tidy3d-mcp` and forwards all remaining command-line arguments:

```bash
uvx tidy3d mcp
```

If you install the standalone runtime package directly, its equivalent command is:

```bash
tidy3d-mcp
```

For raw MCP client config, run the same command:

```json
{
  "mcpServers": {
    "tidy3d": {
      "command": "uvx",
      "args": ["tidy3d", "mcp"]
    }
  }
}
```


## Issues / Feedback / Bug Reporting

Your feedback helps us immensely!

If you find bugs, file an [Issue](https://github.com/flexcompute/tidy3d/issues).
For more general discussions, questions, comments, anything else, open a topic in the [Discussions Tab](https://github.com/flexcompute/tidy3d/discussions).

## License

[GNU LGPL](https://github.com/flexcompute/tidy3d/blob/develop/LICENSE)
