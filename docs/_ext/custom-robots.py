from __future__ import annotations

import os


def process_robots_txt(app, exception):
    # Get the path to the robots.txt file
    robots_file = os.path.join(app.outdir, "robots.txt")
    with open(robots_file) as f:
        contents = f.read()

    site_map = "/".join(
        [app.config["html_baseurl"], app.config["language"], app.config["version"], "sitemap.xml"]
    ).replace("//", "/")
    contents += f"\nSitemap: {site_map}"

    # Update the robots.txt file with the modified contents
    with open(robots_file, "w") as f:
        f.write(contents)


def setup(app):
    # Bind the process_sitemap function to build-finished event
    # exclude_pattern= dir(app.config)
    app.connect("build-finished", process_robots_txt)
