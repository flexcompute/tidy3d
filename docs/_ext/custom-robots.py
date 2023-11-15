import os


def process_robots_txt(app, exception):
    # Get the path to the source directory
    srcdir = app.builder.srcdir

    # Get the path to the robots.txt file
    robots_file = os.path.join(app.outdir, "robots.txt")

    # Read the contents of the robots.txt file
    with open(robots_file) as f:
        contents = f.read()

    # Modify the contents as needed
    # site_map = app.config['html_baseurl'] + app.config['version'] + app.config['language'] + 'sitemap.xml'
    site_map = "/".join(
        [app.config["html_baseurl"], app.config["language"], app.config["version"], "sitemap.xml"]
    ).replace("//", "/")
    new_content = f"\nSitemap: {site_map}"
    contents += new_content

    # Update the robots.txt file with the modified contents
    with open(robots_file, "w") as f:
        f.write(contents)


def setup(app):
    # Bind the process_sitemap function to build-finished event
    # exclude_pattern= dir(app.config)
    app.connect("build-finished", process_robots_txt)
