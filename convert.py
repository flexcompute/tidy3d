from subprocess import run, CalledProcessError
from bs4 import BeautifulSoup
import nbformat
import yaml
import re
import glob
import os
import shutil
from nbconvert import HTMLExporter

output_dir = "./html"
css_file_name = "example-notebook.css"


def creat_yaml(meta, anchor={}):
    data = {
        "layout": "example",
        "custom_css": "cobalt",
        "custom_font": "font2",
        "source_link": meta["source_link"],
    }
    if "title" in meta:
        data["title"] = meta["title"]
    if "description" in meta:
        data["description"] = meta["description"]
    if "keywords" in meta:
        data["tags"] = [tag.strip() for tag in meta["keywords"].split(",")]
    if "page_title" in meta:
        data["page_title"] = meta["page_title"]
    if "image" in meta:
        data["image"] = meta["image"]
    anchor_str = ""
    if anchor:
        anchor_str = yaml.dump(anchor)

    return "---\n" + yaml.dump(data) + anchor_str + "---\n"


def read_template():
    with open(f"./_template/template.html", "r") as f:
        html = f.read()
        return BeautifulSoup(html, "html.parser")


def get_anchor_list(soup):
    """
    h1, h2, h3 can be used to anchor
    """
    anchor_dict = {"anchor_area": {}}
    h1 = soup.find("h1")
    if h1:
        anchor_dict["anchor_area"]["key"] = h1.get("id")
        text = h1.get_text(strip=True)
        for child in h1.children:
            if child.name is not None:
                text = text.replace(child.get_text(strip=True), "")
        anchor_dict["anchor_area"]["label"] = text
    h2 = soup.find_all("h2")
    if h2:
        anchor_dict["anchor_area"]["list"] = []
        for tag in h2:
            dict = {}
            dict["key"] = tag.get("id")
            text = tag.get_text(strip=True)
            for child in tag.children:
                if child.name is not None:
                    text = text.replace(child.get_text(strip=True), "")
            dict["label"] = text
            anchor_dict["anchor_area"]["list"].append(dict)
    return anchor_dict


def write_css_file(style_tags, output_dir):
    css_content = ""
    for style_tag in style_tags:
        css_content += style_tag.string

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + css_file_name, "w+") as f:
        f.write(css_content)


def write_template(meta, file_name, create_css=False):
    html_output_file = f"{output_dir}/{file_name}.html"
    css_output_directory = f"{output_dir}/css/"
    with open(html_output_file, "r") as html_file:
        content = html_file.read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        # anchor = get_anchor_list(soup)
        style_tags = soup.find_all("style")
        for tag in soup.find_all("style"):
            tag.string = re.sub(r"body\s*{\s*(.*?)\s*}", "", tag.string, flags=re.DOTALL)
            tag.string = re.sub(r"html\s*{\s*(.*?)\s*}", "", tag.string, flags=re.DOTALL)
            # remove *
            # tag.string = re.sub(r'\*\s*{\s*(.*?)\s*}', '', tag.string, flags=re.DOTALL)
            tag.string = re.sub(r", \*|,\*", "", tag.string, flags=re.DOTALL)
            # remove *::before
            # tag.string = re.sub(r'\*\s*::before\s*{\s*(.*?)\s*}', '', tag.string, flags=re.DOTALL)
            tag.string = re.sub(r"\*\:\:before\,", "", tag.string, flags=re.DOTALL)
            # remove *::after
            tag.string = re.sub(r"\*\s*::after\s*{\s*(.*?)\s*}", "", tag.string, flags=re.DOTALL)
        if create_css:
            write_css_file(style_tags, css_output_directory)

        # css_file_name = file_name+'.css'
        css_output_file = "" + css_output_directory + css_file_name
        if os.path.exists(css_output_file) and os.path.isfile(css_output_file):
            link_tag = soup.new_tag("link")
            link_tag["rel"] = "stylesheet"
            link_tag["href"] = f"/assets/tidy3d/examples/css/{css_file_name}"

            head_tag = soup.head
            head_tag.insert(4, link_tag)

            for tag in soup.find_all("style"):
                tag.extract()

        for script in soup.head.find_all("script"):
            if "require.min.js" in str(script):
                script.decompose()

        # template_soup = read_template()
        # insert_location = template_soup.find('main', {'id': 'notebook-container'})

        # insert_location.append(soup)

        # remove title tag
        title_tag = soup.find("title")
        if title_tag:
            title_tag.decompose()

        ouptut_html = str(soup)
        # remove <!DOCTYPE html>
        pattern = re.compile(r"<!DOCTYPE html>")
        ouptut_html = pattern.sub("", ouptut_html)

        ouptut_html = ouptut_html.replace("In [", "[")
        yam_str = creat_yaml(meta)
        with open(html_output_file, "w") as output_file:
            output_file.write(yam_str + str(ouptut_html))


def generatorOG(metadata, description):
    feature_image = metadata.get("feature_image", "")
    img_output_directory = f"{output_dir}/image/"
    if feature_image:
        image_name = os.path.basename(feature_image)
        if not os.path.exists(img_output_directory):
            os.makedirs(img_output_directory)

        shutil.copy(feature_image, img_output_directory)
        return {"path": f"/assets/tidy3d/examples/image/{image_name}", "alt": description}

    return None


ipynb_files = glob.glob("*.ipynb")
shutil.rmtree(output_dir)

os.mkdir(output_dir)
index = 0
for input_file in ipynb_files:
    if input_file.endswith(".ipynb"):
        print(f"\033[1m [Converting] : {input_file}\033[0m")

        try:
            cmd = (
                f"jupyter nbconvert --to html --output-dir {output_dir} {input_file} --embed-images"
            )

            result = run(cmd, shell=True, capture_output=True, text=True, check=True)

            if result.returncode == 0:
                with open(input_file, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                    default_title = ""
                    for cell in nb.cells:
                        if cell.cell_type == "markdown" and cell.source.startswith("#"):
                            _title = cell.source.strip("#").strip()
                            default_title = _title.split("\n")[0]
                            break
                    html_exporter = HTMLExporter()
                    # Modify the HTML title tag to use the notebook's metadata
                    metadata = nb["metadata"]
                    title = metadata.get("title", default_title)
                    description = metadata.get("description", "")
                    keywords = metadata.get("keywords", "")
                    dict = {
                        "page_title": default_title,
                        "source_link": f"https://docs.flexcompute.com/projects/tidy3d/en/latest/_sources/notebooks/{input_file}",
                    }
                    open_graph = generatorOG(metadata, title if title else default_title)
                    if title:
                        dict["title"] = title
                    if description:
                        dict["description"] = description
                    if keywords:
                        dict["keywords"] = keywords
                    if open_graph:
                        dict["image"] = open_graph
                    write_template(
                        dict, os.path.splitext(os.path.basename(input_file))[0], index == 0
                    )
                    index += 1

            else:
                print("Failed to convert notebook to HTML.")
        except CalledProcessError as e:
            print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}: {e.stderr}")
            break
