********************
Tidy3D + AI |:bulb:|
********************

*Ushering in a new era of AI-assisted photonic design*

We are pleased to announce the release of the new Tidy3D Extension, which enables seamless building, visualization, and iteration on electromagnetic simulations within Visual Studio Code or Cursor. The extension automatically detects simulations in Tidy3D Python scripts and notebooks, opens an interactive 3D viewer alongside your code, and enables the IDE AI assistant to leverage the integrated FlexAgent MCP server for an intelligent, physics‑aware assistance experience.

.. image:: ../_static/img/tidy3d_extension.png
   :alt: Tidy3D Extension
   :height: 300px

Key Components
--------------

FlexAgent MCP
~~~~~~~~~~~~~

`FlexAgent MCP <flex_agent.html>`_ is a physics-aware AI assistant that understands Tidy3D workflows and electromagnetic simulation concepts. Unlike generic coding assistants, FlexAgent provides specialized knowledge and can control the 3D Viewer to provide comprehensive, context-aware support.

**Key Capabilities:**

- **Natural Language Interaction** – Build, modify, and analyze simulations using plain English
- **Physics-Aware** – Understands electromagnetic simulation concepts and Tidy3D's API
- **Learning & Troubleshooting** – Get explanations and fix simulation issues
- **Code Generation** – Create complete simulation setups from scratch
- **Result Analysis** – Interpret and visualize simulation data

3D Viewer
~~~~~~~~~

The `3D Viewer <3d_viewer.html>`_ provides interactive, real-time visualization of Tidy3D simulations directly within your IDE. It automatically detects simulation objects in your code and opens alongside your editor, providing immediate visual feedback as you develop.

**Key Features:**

- **Automatic Detection** – Identifies ``Simulation`` objects and opens automatically
- **Live Synchronization** – Updates instantly when you modify code
- **Interactive Navigation** – Rotate, zoom, and pan to explore simulations
- **AI-Controlled** – FlexAgent can navigate and explain the 3D scene
- **Comprehensive Visualization** – View structures, sources, monitors, and boundaries

Video Demos
-----------

Watch these demonstration videos to see Tidy3D + AI capabilities in action. Each video showcases specific workflows and features that highlight how FlexAgent MCP and the 3D Viewer can accelerate your photonic design process.

- **Mode Analysis** – `Learn how to use AI assistance to perform mode analysis and understand waveguide modes <https://youtu.be/dQMGgQr_6KE?si=fUXlMEjk2RHCEWjF>`_

- **Data Analysis** – `See how FlexAgent helps analyze simulation results and extract meaningful insights from your data <https://youtu.be/CLFoePrHcjM?si=j7izF3RF1rV7HpO_>`_

- **Convergence Test** – `Discover how to set up and run convergence tests with AI guidance to ensure simulation accuracy <https://youtu.be/fhfi4U2HKbo?si=rLo9Fgg-Zv-pJUKV>`_

- **Cost Estimation** – `Understand how to estimate simulation costs and optimize your computational resources with AI assistance <https://youtu.be/ZuZVu7yA6DQ?si=YNWn9RB3Thmph62a>`_

For more comprehensive tutorials and the complete video series, visit our `Tidy3D + AI Video Playlist <https://www.youtube.com/playlist?list=PL7kxN4u_N9HHb1QBPXhTlYEMjIt2SY1re>`_.

Using Tidy3D with AI Coding Agents
-----------------------------------

The FlexAgent MCP works with any MCP-compatible AI coding agent — not just the Tidy3D IDE extensions. To get started:

1. **Set up the MCP** — follow the instructions in `FlexAgent MCP — Standalone MCP Client <flex_agent.html#standalone-mcp-client>`_ for your client (Claude Code, Codex CLI, Cursor, Gemini CLI, etc.).
2. **Add a bootstrap file** — save the markdown below as ``AGENTS.md`` (or ``CLAUDE.md`` for Claude Code) in your project root directory. This tells the agent how to use the MCP effectively.
3. **Review the simulation tips** — the `Simulation Tips for AI Agents <simulation_tips.html>`_ page covers workflow patterns, parameter sweeps, inverse design, and common pitfalls.

.. code-block:: markdown

   # Tidy3D Agent Instructions

   ## MCP Setup

   If the Tidy3D MCP server is not already configured, see:
   https://docs.flexcompute.com/projects/tidy3d/en/latest/ai/flex_agent.html#standalone-mcp-client

   ## Workflow

   Use `search_flexcompute_docs` and `fetch_flexcompute_doc` from the Tidy3D MCP as
   your primary documentation source throughout the session. Before writing simulation
   code:

   1. Search for "Simulation Tips for AI Agents" — workflow patterns, pitfalls,
      and checklists.
   2. Search for the relevant class docstrings — they contain selection guides,
      extraction patterns, and practical advice.
   3. Search for example notebooks related to the device or workflow you are building.


.. toctree::

    flex_agent
    3d_viewer
    simulation_tips
    cursor_extension
    vscode_extension


