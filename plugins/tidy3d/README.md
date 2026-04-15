# Tidy3D Codex Plugin

This directory is the local-plugin bundle for installing Tidy3D in Codex through a personal marketplace. Installing it registers `tidy3d-mcp` as an MCP server so Codex can search Tidy3D documentation and assist with simulations.

## Requirements

- `git`
- `uv`

## Install instructions

**1. Clone the repository**

```bash
git clone https://github.com/flexcompute/tidy3d.git
```

**2. From the root of the repository, copy the plugin bundle to the Codex plugin directory in your home directory.**

```bash
cp -rL ./plugins/tidy3d ~/.codex/plugins/tidy3d
```

`-rL` dereferences symlinks so the installed copy contains plain files.

**3. Register the plugin in your home directory in a personal marketplace.**

If `~/.agents/plugins/marketplace.json` does not exist, create it and insert

```json
{
  "name": "tidy3d-plugins",
  "interface": { "displayName": "Tidy3D Plugins" },
  "plugins": [{
    "name": "tidy3d",
    "source": {
      "source": "local",
      "path": "./.codex/plugins/tidy3d"
    },
    "policy": {
      "installation": "AVAILABLE",
      "authentication": "ON_INSTALL"
    }
  }]
}
```

**4. Restart Codex**.

The plugin should appear in the Codex plugin list under the name **Tidy3D**.

## Uninstall

```bash
rm -rf ~/.codex/plugins/tidy3d
```

Then remove the `tidy3d` entry from `~/.agents/plugins/marketplace.json` and restart Codex.
