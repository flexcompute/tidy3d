from __future__ import annotations

import os


def process_robots_txt(app, exception):
    # Get the path to the robots.txt file
    robots_file = os.path.join(app.outdir, "robots.txt")
    with open(robots_file) as f:
        contents = f.read()

    site_map = f"{app.config['html_baseurl'].rstrip('/')}/en/latest/sitemap.xml"
    contents += f"\nSitemap: {site_map}\n"
    contents += "Disallow: /projects/tidy3d/en/v*/\n"

    # Update the robots.txt file with the modified contents
    with open(robots_file, "w") as f:
        f.write(contents)


def setup(app):
    # Bind the process_sitemap function to build-finished event
    # exclude_pattern= dir(app.config)
    app.connect("build-finished", process_robots_txt)
