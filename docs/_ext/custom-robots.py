from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit


def process_robots_txt(app, exception):
    # Get the path to the robots.txt file
    robots_file = os.path.join(app.outdir, "robots.txt")
    with open(robots_file) as f:
        contents = f.read()

    html_baseurl = app.config["html_baseurl"].rstrip("/")
    parts = urlsplit(html_baseurl)
    path = parts.path.rstrip("/")
    segments = [segment for segment in path.split("/") if segment]
    if (
        len(segments) >= 2
        and segments[-2] == "en"
        and (segments[-1] == "latest" or segments[-1].startswith("v"))
    ):
        segments = segments[:-2]
    base_path = f"/{'/'.join(segments)}" if segments else ""
    base_url = urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))
    site_map = f"{base_url}/en/latest/sitemap.xml"

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
