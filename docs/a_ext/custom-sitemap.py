import xml.etree.ElementTree as ET
import re


def match_exclude_url(app, url):
    exclude_pattern = app.config._raw_config["custom_sitemap_excludes"]
    for regex in exclude_pattern:
        if re.search(regex, url):
            return True
    return False


def process_sitemap(app, exception):
    # Only process sitemap for successful builds
    if exception is not None:
        return

    # Get path to sitemap.xml file
    sitemap_path = app.outdir + "/sitemap.xml"

    # Parse sitemap.xml file
    tree = ET.parse(sitemap_path)
    root = tree.getroot()

    # Process each URL node in the sitemap
    for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
        loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
        if match_exclude_url(app, loc):
            root.remove(url)
        # Do something with the URL...
        with open(sitemap_path, "wb") as f:
            tree.write(f, encoding="UTF-8")


def setup(app):
    # Bind the process_sitemap function to build-finished event
    app.connect("build-finished", process_sitemap)
