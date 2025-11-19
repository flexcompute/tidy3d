3D Viewer
=========

*Interactive visualization of Tidy3D simulations directly within your IDE*

The 3D Viewer is an interactive visualization tool that provides real-time, three-dimensional rendering of Tidy3D simulations directly within your code editor. Integrated seamlessly into the Tidy3D extensions for `Cursor <cursor_extension.html>`_ and `VS Code <vscode_extension.html>`_, the viewer automatically detects simulation objects in your Python scripts and notebooks, opening alongside your code to provide immediate visual feedback as you develop.

Overview
--------

The 3D Viewer transforms how you interact with electromagnetic simulations by bringing visualization directly into your development environment. Instead of switching between your code editor and external visualization tools, you can see your simulation geometry, sources, monitors, and boundary conditions update in real-time as you modify your code.

.. image:: ../_static/img/viewer.png
   :alt: Tidy3D 3D Viewer
   :height: 300px

Key Features
------------

- **Automatic Detection** – Automatically identifies ``Simulation`` objects in your code and opens the viewer
- **Live Synchronization** – Updates instantly when you modify geometry, materials, sources, or monitors
- **Interactive Navigation** – Rotate, zoom, and pan to explore your simulation from any angle
- **AI-Controlled** – `FlexAgent MCP <flex_agent.html>`_ can navigate and explain the 3D scene through natural language
- **Editor-Native** – No context switching; visualization happens right where you code
- **Comprehensive Visualization** – View structures, sources, monitors, boundary conditions, and simulation domains

Getting Started
---------------

The 3D Viewer is automatically available when you install the Tidy3D extension for `Cursor <cursor_extension.html>`_ or `VS Code <vscode_extension.html>`_. No additional setup is required.

1. **Create or open a Tidy3D simulation** in your Python script or Jupyter notebook.
2. **Define a ``Simulation`` object** – The viewer automatically detects it.
3. **Open Viewer** – Click the status bar, use the command palette (``Cmd/Ctrl+Shift+P -> Tidy3D: Open Tidy3D``), or ask the AI assistant to open the viewer.

The 3D Viewer is part of the comprehensive Tidy3D + AI ecosystem. For more information, see the `Tidy3D + AI overview <index.html>`_.

