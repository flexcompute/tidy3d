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

    html_baseurl = f"{app.config['html_baseurl'].rstrip('/')}/"
    if html_baseurl.startswith("https://dev.docs.flexcompute.com/"):
        contents = "User-agent: *\nDisallow: /\n"
        with open(robots_file, "w") as f:
            f.write(contents)
        return

    expected_baseurls = {"https://docs.flexcompute.com/projects/tidy3d/en/latest/"}
    default_sitemap_baseurl = "https://docs.flexcompute.com/projects/tidy3d/en/latest/"
    rtd_version = os.environ.get("TIDY3D_DOCS_VERSION") or os.environ.get("READTHEDOCS_VERSION")
    if rtd_version == "latest":
        if html_baseurl not in expected_baseurls:
            raise ValueError(
                "html_baseurl must be a supported latest docs URL for robots.txt "
                "generation. "
                f"Expected one of {sorted(expected_baseurls)!r}, got {html_baseurl!r}."
            )
        site_map = f"{html_baseurl.rstrip('/')}/sitemap.xml"
    else:
        site_map = f"{default_sitemap_baseurl.rstrip('/')}/sitemap.xml"

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
