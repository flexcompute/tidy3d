Troubleshooting |:tools:|
=========================

.. note::

    If you encounter any technical difficulties or have inquiries related to
    using Tidy3D, kindly submit a Customer Technical Support form, available on
    the right side of the `Tidy3D Technical Support page
    <https://www.flexcompute.com/tidy3d/technical-support/>`__. To ensure
    efficient assistance, please provide a thorough description of the issue or
    question, along with relevant screenshots or a link to a Loom recording.

    As a token of our appreciation, for well-documented and confirmed bug
    reports, we will reward you with up to 50 FlexCredits.

    Running ``tidy3d troubleshoot report`` (documented below) is the quickest
    way to attach a thorough, paste-ready description of your environment and
    connectivity state to the form.

When something is wrong with your Tidy3D setup — a simulation hangs, an upload
fails, or authentication misbehaves — the fastest path to a fix is to run
``tidy3d troubleshoot`` and share the printed report with support. The command
group collects platform, package, configuration, and connectivity information
without exposing your API key.

Quick Start
-----------

.. code-block:: shell

    tidy3d troubleshoot report

Prompts for a short description, when it started, reproducibility, run mode,
and script location, then probes your environment and Tidy3D connectivity and
prints a bundle matching the issue-report template. Add ``--output report.md``
(or ``-o report.md``) to write it to a file instead of stdout.

Subcommands
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Purpose
   * - ``troubleshoot report``
     - Full bundle: prompts + environment + configuration + connectivity.
   * - ``troubleshoot environment``
     - Offline snapshot: Python, platform, Flexcompute packages,
       ``pip freeze``, and the redacted client configuration.
   * - ``troubleshoot connection``
     - Network probe only: DNS, TCP, TLS, API latency, auth, 100 MiB download.

Options
-------

Common to every subcommand:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Effect
   * - ``--output PATH`` / ``-o PATH``
     - Write the report to ``PATH`` instead of stdout (atomic replace).
   * - ``--json``
     - Emit structured JSON instead of the text summary.
   * - ``--verbose``
     - Print each phase's individual probes to stderr before it runs.
   * - ``--private-network-details``
     - Include proxy hosts, ``NO_PROXY``, certificate paths, resolved IPs, and
       TLS certificate metadata. Marks the report ``privacy_mode: private``;
       share only with your IT administrators.

``troubleshoot report`` extras:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Effect
   * - ``--task-id ID``
     - Populates question 5 of the template.
   * - ``--no-connection``
     - Skip live network probes.
   * - ``--traceback-file PATH``
     - Inline the file contents into question 6 (steps to reproduce).
   * - ``--non-interactive``
     - Skip narrative prompts (for scripts / CI).
   * - ``--api-samples N`` / ``--timeout SECONDS``
     - Latency-sample count (default 3) and per-request timeout (default 20s).

Report Layout
-------------

1. **Issue template** — the seven questions, filled from your answers plus
   the auto-detected Tidy3D version and task ID.
2. **Environment** — Python, platform, Flexcompute distributions
   (``tidy3d``, ``tidy3d-extras``, ``flex-rf``, ``photonforge``, ``flow360``),
   and a full ``pip freeze``.
3. **Configuration** — redacted API endpoint, SSL settings, proxy /
   certificate / Tidy3D env vars.
4. **Connection diagnostics** — DNS, TCP + TLS, API latency, auth, and
   storage-download throughput, each with pass/fail and (where relevant) a
   recommended next step.

Privacy
-------

Default output is ``privacy_mode: shareable`` and safe to send to support:
API keys, signed URL secrets, and proxy hosts are redacted or presence-only.
``--private-network-details`` switches to ``privacy_mode: private`` and
embeds internal network details for IT — do not share it externally.

From Python
-----------

The CLI is a thin wrapper over three helpers. Each returns a Pydantic model
with ``support_text()`` and ``model_dump_json()``.

.. code-block:: python

    from tidy3d import web

    print(web.diagnose_environment().support_text())
    print(web.diagnose_connection().support_text())
    print(web.diagnose_report(task_id="fdve-01H8ABCDEXAMPLE").support_text())
