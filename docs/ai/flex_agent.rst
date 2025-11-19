FlexAgent MCP
=============

*AI-powered simulation assistance for Tidy3D through Model Context Protocol*

FlexAgent MCP is a Model Context Protocol (MCP) server that brings physics-aware AI assistance to Tidy3D workflows. Unlike generic coding assistants, FlexAgent understands electromagnetic simulation concepts, Tidy3D's API, and can interact with the `3D Viewer <3d_viewer.html>`_ to provide comprehensive, context-aware support for your simulation projects.

📹 **Watch the demo**: `FlexAgent MCP Introduction Video <https://youtu.be/KU6QP-gqUGA?list=PL7kxN4u_N9HHb1QBPXhTlYEMjIt2SY1re>`_

Overview
--------

FlexAgent enables natural language interaction with Tidy3D simulations across multiple platforms, including `Cursor <cursor_extension.html>`_, `VS Code <vscode_extension.html>`_, and other MCP-compatible environments. Through structured workflows that leverage Tidy3D's trusted documentation, FlexAgent transforms your AI assistant into a knowledgeable simulation partner capable of:

- **Learning** – Understanding Tidy3D concepts and workflows: *"Explain how Tidy3D FDTD updates the fields in this notebook"*
- **Troubleshooting** – Diagnosing and resolving simulation issues: *"I want to fix the structure-related validation error"*
- **Customization** – Modifying existing simulations: *"Replace the FluxMonitor with a ModeMonitor"*
- **Build from Scratch** – Creating new simulation setups: *"Create a y-junction waveguide simulation setup"*
- **Result Analysis** – Interpreting and visualizing simulation data: *"Plot all the electric field components"*

For a comprehensive introduction to the Tidy3D + AI ecosystem, see the `Tidy3D + AI overview <index.html>`_.

Installation
------------

FlexAgent MCP can be installed in two ways: as part of the Tidy3D IDE extensions (recommended) or as a standalone MCP server for use with other MCP-compatible clients.

Tidy3D Extensions
~~~~~~~~~~~~~~~~~

**Recommended for most users**: FlexAgent MCP is seamlessly integrated within the Tidy3D extensions for Cursor and Visual Studio Code. When you install and configure these extensions, FlexAgent is automatically set up and ready to use.

For detailed installation and setup instructions, see:

- `Tidy3D for Cursor <cursor_extension.html>`_ – Complete guide for Cursor users
- `Tidy3D for VS Code <vscode_extension.html>`_ – Complete guide for Visual Studio Code users

The extensions handle all FlexAgent configuration automatically, including API key management, MCP server deployment, and integration with the `3D Viewer <3d_viewer.html>`_.

Standalone MCP Client
~~~~~~~~~~~~~~~~~~~~~

For users who want to use FlexAgent with other MCP-compatible clients (without the full Tidy3D extension), you can install the ``tidy3d-mcp`` server directly.

**Prerequisites:**

- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ package manager installed
- A Tidy3D API key (`get one free here <https://tidy3d.simulation.cloud/signup>`_)

**Register the server with your MCP client** – Use the configuration block below that matches your MCP client:

**Codex CLI / IDE**

.. code-block:: bash

   codex mcp add tidy3d -- uvx tidy3d-mcp --api-key "YOUR_TIDY3D_API_KEY"

**Claude CLI / Desktop / Code**

.. code-block:: bash

   claude mcp add tidy3d -- uvx tidy3d-mcp --api-key "YOUR_TIDY3D_API_KEY"

**Gemini CLI**

Create or edit ``.gemini/settings.json`` (project) or ``~/.gemini/settings.json`` (global):

.. code-block:: json

   {
     "mcpServers": {
       "tidy3d": {
         "command": "uvx",
         "args": ["tidy3d-mcp", "--api-key", "YOUR_TIDY3D_API_KEY"]
       }
     }
   }

**Cursor CLI / IDE**

Cursor reuses the same schema across the editor and ``cursor-agent``. Configure ``.cursor/mcp.json`` (per-project) or ``~/.cursor/mcp.json`` (global) and then run ``cursor-agent mcp list`` to verify:

.. code-block:: json

   {
     "mcpServers": {
       "tidy3d": {
         "command": "uvx",
         "args": ["tidy3d-mcp", "--api-key", "YOUR_TIDY3D_API_KEY"]
       }
     }
   }


AI Collaboration Best Practices
--------------------------------

To get the most out of FlexAgent, follow these proven strategies for effective AI-assisted simulation development:

Leverage FlexAgent Rules
~~~~~~~~~~~~~~~~~~~~~~~~~

**For optimal results**, explicitly reference the integrated FlexAgent rules in your prompts. When using the Tidy3D extensions, these rules are automatically loaded:

- **Cursor**: Rules are in ``.cursor/rules/`` (automatically loaded)
- **VS Code**: Rules are in ``.github/instructions/`` (enable in Copilot settings)

Example: *"Follow the FlexAgent rules in* ``flexagent.mdc`` *for all responses. These rules take priority over other instructions."*

Start Simple, Build Incrementally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Begin with basic simulations and gradually add complexity. This approach makes debugging easier and helps you understand each component before moving to the next level. Ask FlexAgent to explain concepts as you build.

Provide Rich Context
~~~~~~~~~~~~~~~~~~~~

The more specific your prompts, the better FlexAgent's assistance. Instead of generic requests like "help with mode analysis," provide detailed context:

**Better prompt**: *"I'm studying photonic integrated circuits. How can I use Tidy3D to analyze the fundamental TE mode of a 220 nm thick silicon strip waveguide?"*

Use the Viewer for Real-Time Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the `3D Viewer <3d_viewer.html>`_ open while editing your simulation code. Changes to geometry, materials, sources, or monitors update automatically, letting you immediately visualize the impact of parameter tweaks without re-running the code. This creates a powerful feedback loop for rapid iteration.

Be Specific When Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of vague requests like "fix all errors," provide targeted information:

- *"Fix the boundary condition warning for structure[0]"*
- *"Resolve the mesh refinement issue near the waveguide core"*
- *"Clear the validation error related to ModeSolver plane"*

Understand Before Proceeding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take time to comprehend the generated code and simulation setup. Ask FlexAgent to explain concepts you're unfamiliar with before building upon them. This ensures you maintain control and understanding of your simulation.

Use Incremental Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After each major change, ask FlexAgent to *"validate and explain the simulation changes"* before proceeding. This helps ensure physical accuracy and catches issues early.

Document Your Simulation Intent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add comments explaining the physics you're trying to model. This helps both you and FlexAgent understand the simulation goals:

.. code-block:: python

   # Modeling evanescent coupling between two strip waveguides separated by 200 nm

Specify Your Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with multiple Python environments, explicitly tell FlexAgent which one to use:

*"Run this code using my* ``tidy3d_latest`` *conda environment located at* ``/home/user/miniconda3/envs/tidy3d_latest``."

Prompt Examples
---------------

The following examples demonstrate effective ways to interact with FlexAgent across different use cases. These prompts leverage FlexAgent's understanding of Tidy3D workflows and its ability to access documentation and control the `3D Viewer <3d_viewer.html>`_.

Learning
~~~~~~~~

Use FlexAgent to understand Tidy3D concepts, methods, and best practices:

- *"How to set up a GaussianBeam source"*
- *"Explain how the FDTD method works"*
- *"How can I perform a mode analysis?"*
- *"What's the difference between a FluxMonitor and a ModeMonitor?"*
- *"Explain the relationship between mesh refinement and simulation accuracy"*

Troubleshooting
~~~~~~~~~~~~~~~

Get help diagnosing and fixing simulation issues:

- *"Fix the warning related to 'structures[0]' extending exactly to simulation edges"*
- *"Fix the code to avoid the warning related to Bloch vector"*
- *"Clear the validation error related to ModeSolver plane"*
- *"Why is my simulation diverging, and how do I fix it?"*
- *"Help me resolve the boundary condition warning for my periodic structure"*

Customization
~~~~~~~~~~~~~

Modify existing simulations to explore different configurations:

- *"Add a mode source at the position (-4, 0, 0) to inject the fundamental TE waveguide mode at 1.55 micrometers"*
- *"Build a mode simulation setup to analyze the dispersion parameters of the first 2 optical modes at the source plane of my FDTD simulation setup"*
- *"Perform a parameter sweep to vary the unit cell period between 0.3 to 0.5 micrometers"*
- *"Replace the FluxMonitor with a ModeMonitor and update the analysis code"*
- *"Change the simulation boundary conditions from PML to periodic in the x-direction"*

Build From Scratch
~~~~~~~~~~~~~~~~~~

Create complete simulation setups from natural language descriptions:

- *"Create a simple FDTD simulation setup to simulate a silicon strip waveguide with dimensions (0.5, 0.22, 10) and refractive index of 3.47"*
- *"I am interested in PIC devices. Build a setup to obtain the modal power at the outputs of a y-junction waveguide. Add s-bends to the output waveguides"*
- *"I want to build a simulation setup to investigate structural color generation"*
- *"Create a ring resonator simulation with a 5 μm radius and 220 nm waveguide thickness"*
- *"Build a metasurface simulation for beam steering at 1.55 μm wavelength"*

Analysis
~~~~~~~~

Interpret results and create visualizations:

- *"Plot the field intensity profile obtained from the FieldMonitor at the central frequency"*
- *"Considering my parameter sweep simulation, plot the |E|^2 field profile for all the modes in the largest waveguide"*
- *"Plot the reflectance spectrum of all unit cell period values in the parameter sweep simulation in a single chart"*
- *"Analyze the transmission spectrum and identify the resonance frequencies"*
- *"Create a 2D plot showing the electric field distribution in the x-y plane at z=0"*

