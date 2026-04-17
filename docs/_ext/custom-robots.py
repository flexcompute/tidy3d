from __future__ import annotations

import os


def process_robots_txt(app, exception):
    if exception is not None:
        return

    # Get the path to the robots.txt file
    robots_file = os.path.join(app.outdir, "robots.txt")
    if not os.path.exists(robots_file):
        return

    with open(robots_file) as f:
        contents = f.read()

    expected_baseurl = "https://docs.flexcompute.com/projects/tidy3d/en/latest/"
    html_baseurl = f"{app.config['html_baseurl'].rstrip('/')}/"
    rtd_version = os.environ.get("READTHEDOCS_VERSION")
    if rtd_version == "latest" and html_baseurl != expected_baseurl:
        raise ValueError(
            "html_baseurl must be the latest docs URL for robots.txt generation. "
            f"Expected {expected_baseurl!r}, got {html_baseurl!r}."
        )
    site_map = f"{expected_baseurl.rstrip('/')}/sitemap.xml"

    lines = [
        line
        for line in contents.splitlines()
        if not line.startswith("Sitemap:")
        and line.strip().lower() != "disallow: /projects/tidy3d/en/v*/"
    ]
    inserted = False
    for index, line in enumerate(lines):
        if line.lower().startswith("user-agent:"):
            lines.insert(index + 1, "Disallow: /projects/tidy3d/en/v*/")
            inserted = True
            break
    if not inserted:
        lines.append("User-agent: *")
        lines.append("Disallow: /projects/tidy3d/en/v*/")
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(f"Sitemap: {site_map}")
    contents = "\n".join(lines) + "\n"

    # Update the robots.txt file with the modified contents
    with open(robots_file, "w") as f:
        f.write(contents)


def setup(app):
    # Bind the process_sitemap function to build-finished event
    # exclude_pattern= dir(app.config)
    app.connect("build-finished", process_robots_txt)
    return {"parallel_read_safe": True}
