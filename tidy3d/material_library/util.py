from io import StringIO
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..components.viz import FLEXCOMPUTE_COLORS

MAX_POLES_TO_DISPLAY = 3


def summarize_material_library(matlib) -> str:
    """Return a user-friendly multiline string summarizing the entire material library."""
    lines = ["Material Library Summary:"]
    for key, mat_item in sorted(matlib.items()):
        if key != "graphene":
            lines.append(
                f"  - Key: {key}, Name: {mat_item.name}, "
                f"Default Variant: {mat_item.default}, "
                f"# Variants: {len(mat_item.variants)}"
            )
        else:
            lines.append(f"  - Key: {key}")
    return "\n".join(lines)


def summarize_material_item(m) -> str:
    """
    Return a concise summary of a single MaterialItem instance:
    includes name, default variant, plus the list of all variants.
    """
    lines = [f"Material Summary: {m.name}"]
    lines.append(f"  Default Variant: {m.default}")
    lines.append("  Variants:")
    for vname in m.variants.keys():
        lines.append(f"    - {vname}")
    return "\n".join(lines)


def variant_name(v):
    """Retrieve variant name for display."""
    default_name = "Unnamed Variant"
    name = default_name
    # Check attributes safely
    if hasattr(v, "medium") and hasattr(v.medium, "name"):
        name = v.medium.name
    elif hasattr(v, "ordinary") and hasattr(v.ordinary, "name"):
        name = v.ordinary.name
    # Ensure name is not None if getattr found the attr but it was None
    name = name or default_name

    return name


def summarize_medium(med) -> List[str]:
    """Returns relevant medium information for display."""
    lines = []

    if hasattr(med, "eps_inf"):
        lines.append(f"      eps_inf: {med.eps_inf}")
    if hasattr(med, "poles"):
        lines.append(f"      # poles: {len(med.poles)}")
        # optionally show some truncated details of each pole
        for i, (a, c) in enumerate(med.poles[:MAX_POLES_TO_DISPLAY]):
            lines.append(f"        - pole {i}: a={a}, c={c}")
        if len(med.poles) > MAX_POLES_TO_DISPLAY:
            lines.append("        ... (more poles omitted)")

    # or if MultiPhysicsMedium or other types, handle accordingly
    if hasattr(med, "frequency_range"):
        lines.append(f"      frequency_range: {med.frequency_range}")

    return lines


def summarize_variant_item(v) -> str:
    """
    Return a concise summary for a single VariantItem instance,
    highlighting references and main model parameters (e.g., eps_inf, # poles, freq. range).
    """
    name = variant_name(v)
    lines = []
    lines.append(f"Variant Summary: {name}")

    # references
    if v.reference:
        lines.append("  References:")
        for ref in v.reference:
            if ref.doi:
                lines.append(f"    DOI: {ref.doi}")
            elif ref.journal:
                lines.append(f"    Journal: {ref.journal}")
    else:
        lines.append("  References: None")

    lines.append(f"  data_url: {v.data_url or 'N/A'}")

    lines.append("  Medium model(s):")

    for medium_key, med in v.summarize_mediums.items():
        lines.append(f"    {medium_key}")

        lines += summarize_medium(med)

    return "\n".join(lines)


def get_hex_formatted_string(color_key):
    """Return a hex formatted string from the FLEXCOMPUTE_COLORS that can be used with rich printing."""
    return f"#{FLEXCOMPUTE_COLORS[color_key]:06x}"


def repr_pretty_with_rich(obj, p, cycle):
    """Enable _repr_pretty_ for jupyter notebooks to use the rich printing output."""
    if cycle:
        p.text("MaterialLibrary(...)")
    else:
        # Render the __rich__ object to a string using a fake console
        sio = StringIO()
        console = Console(file=sio, force_terminal=True, width=100)
        console.print(obj.__rich__())
        p.text(sio.getvalue())


def add_medium_details_to_tree(medium, medium_node: Tree):
    """Adds details of a medium dictionary to a Rich Tree node."""

    if hasattr(medium, "eps_inf"):
        medium_node.add(f"[bold]eps_inf:[/bold] {medium.eps_inf}")
    if hasattr(medium, "poles"):
        num_poles = len(medium.poles)
        poles_node = medium_node.add(f"[bold]Poles: {num_poles}[/bold]")

        for i, (a, c) in enumerate(medium.poles[:MAX_POLES_TO_DISPLAY]):
            poles_node.add(f"pole {i}: a={a}, c={c}")
        if num_poles > MAX_POLES_TO_DISPLAY:
            poles_node.add("... (more poles omitted)")

    if hasattr(medium, "frequency_range"):
        medium_node.add(f"[bold]Freq. Range:[/bold] {medium.frequency_range}")


def summarize_material_library_rich(matlib):
    """Returns a Rich Table summarizing the MaterialLibrary."""

    table = Table(
        title=f"[bold {get_hex_formatted_string('brand_black')}]Material Library Summary[/]",
        show_header=True,
        header_style=f"bold {get_hex_formatted_string('brand_black')}",
        border_style=f"{get_hex_formatted_string('brand_black')}",
        expand=False,  # Don't force table to full terminal width
    )

    table.add_column(
        "Key", style=f"{get_hex_formatted_string('brand_blue')}", width=18, justify="center"
    )
    table.add_column(
        "Name", style=f"{get_hex_formatted_string('brand_purple')}", min_width=20, justify="center"
    )
    table.add_column(
        "Default Variant",
        style=f"{get_hex_formatted_string('brand_green')}",
        min_width=20,
        justify="center",
    )
    table.add_column(
        "# Variants", style=f"{get_hex_formatted_string('brand_green')}", justify="center"
    )

    # Iterate through the sorted items for consistent order
    for key, mat_item in sorted(matlib.items()):
        # Special handling for the 'graphene' key
        if key == "graphene":
            table.add_row(
                key,
                "Graphene",
                "---",  # Leave other cells blank for graphene
                "---",
            )
        else:
            name = mat_item.name
            default = mat_item.default if mat_item.default is not None else "[italic]None[/italic]"
            # Safely get length of variants (assuming it's a collection)
            num_variants = (
                str(len(mat_item.variants))
                if hasattr(mat_item, "variants") and mat_item.variants is not None
                else "[red]N/A[/red]"
            )

            table.add_row(key, name, default, num_variants)

    return table


def summarize_material_item_rich(m):
    """Returns a Rich Tree representation summarizing the MaterialItem."""
    tree_title = f"[bold {get_hex_formatted_string('brand_purple')}]Material Summary: {m.name}[/]"
    tree = Tree(tree_title)

    # Add a subtree for the variants
    variants_node = tree.add("[bold]Variants:[/bold]")
    if m.variants:
        # Sort keys for consistent display order
        sorted_variant_names = sorted(m.variants.keys())
        for vname in sorted_variant_names:
            # Add each variant name as a child node.
            variants_node.add(f"{vname}")
    else:
        # Indicate if there are no variants
        variants_node.add("[italic]None[/italic]")

    return Panel(tree, border_style=f"{get_hex_formatted_string('brand_purple')}", expand=False)


def summarize_variant_item_rich(v):
    """Returns Rich renderables for displaying a VariantItem summary."""

    name = variant_name(v)

    tree_title = Text(
        f"Variant Summary: {name}", style=f"bold {get_hex_formatted_string('brand_green')}"
    )
    tree = Tree(tree_title)

    ref_node = tree.add("[bold]References[/bold]")
    if v.reference:
        for ref in v.reference:
            if hasattr(ref, "doi") and ref.doi:
                ref_node.add(f"[bold]DOI:[/bold] [link=https://doi.org/{ref.doi}]{ref.doi}[/link]")
            elif hasattr(ref, "journal") and ref.journal:
                ref_node.add(f"[bold]Journal:[/bold] {ref.journal}")
            else:
                ref_node.add("[italic]Reference details missing[/italic]")
    else:
        ref_node.add("[italic]None[/italic]")

    data_url_str = f"[bold]data_url:[/bold] {v.data_url or '[italic]N/A[/italic]'}"
    if (
        v.data_url
        and isinstance(v.data_url, str)
        and v.data_url.startswith(("http://", "https://"))
    ):
        data_url_str = f"[bold]data_url:[/bold] [link={v.data_url}]{v.data_url}[/link]"
    tree.add(data_url_str)

    mediums_node = tree.add("[bold]Medium model(s)[/bold]")

    mediums = v.summarize_mediums
    if mediums:
        for med_key, med in mediums.items():
            medium_type_node = mediums_node.add(
                f"[italic {get_hex_formatted_string('brand_blue')}]{med_key}[/]"
            )
            add_medium_details_to_tree(med, medium_type_node)
    else:
        mediums_node.add("[italic]None[/italic]")

    return Panel(tree, border_style=f"{get_hex_formatted_string('brand_green')}", expand=False)
