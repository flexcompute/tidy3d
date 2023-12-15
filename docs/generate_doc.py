# Generate documentation for Material Library (Python)

import numpy as np
from tidy3d import material_library as lib
from tidy3d.constants import C_0
from tidy3d import Medium2D, PoleResidue

LOW_LOSS_THRESHOLD = 2e-5


def generate_material_library_doc():

    # Display unit for "Valid range" in table, select from 'eV', 'um', 'THz'
    unit = "um"

    # doc file path
    fname = "./api/material_library.rst"

    def num2str(num):
        if np.isinf(num):
            return " "
        else:
            return str(round(num, 2))

    with open(fname, "w") as f:

        # Write file header
        header = (
            "****************\n"
            "Material Library\n"
            "****************\n\n"
            ".. currentmodule:: tidy3d\n\n"
            "The material library is a dictionary containing various dispersive models from real world materials. To use the materials in the library, import it first by:\n\n"
            ">>> from tidy3d import material_library\n\n"
            "The key of the dictionary is the abbreviated material name.\n\n"
            'Note: some materials have multiple variant models, in which case the second key is the "variant" name.\n\n'
            'To import a material "mat" of variant "var":\n\n'
            ">>> medium = material_library['mat']['var']\n\n"
            "For example, silver measured by A. D. Rakic et al. (1998) can be loaded as:\n\n"
            ">>> silver = material_library['Ag']['Rakic1998BB']\n\n"
            "You can also import the default variant of a material by:\n\n"
            ">>> medium = material_library['mat'].medium\n\n"
            "It is often useful to see the full list of variants for a given medium:\n\n"
            ">>> print(material_library['mat'].variants.keys())\n\n"
            "To access the details of a variant, including material model, references and tabulated data, use the following command:\n\n"
            ">>> material_library['mat'].variants['var']\n\n\n"
        )
        f.write(header)

        for abbr, mat in sorted(
            lib.items(), key=lambda item: item[0].lower()
        ):  # iterate materials sorted by material abbreviation

            if isinstance(mat, type):
                # Write material title
                title = mat.__name__ + ' ("' + abbr + '")'
                f.write(title + "\n")
                f.write("=" * len(title) + "\n\n")
                f.write(".. autosummary::\n")
                f.write("   :toctree: _autosummary/\n\n")
                f.write("   " + mat.__name__ + "\n\n")

            else:
                # Write material title
                title = mat.name + ' ("' + abbr + '")'
                f.write(title + "\n")
                f.write("=" * len(title) + "\n\n")

                # Place holders
                ref_list = []  # references
                code_string = ""  # example code

                # Initialize table
                columns = ["variant", "range", "model", "ref"]  # column key
                name = {  # column label
                    "variant": "Variant",
                    "range": "Valid for",
                    "model": "Model Info",
                    "ref": "Reference",
                }
                rows = []  # rows of contents
                width = {}  # column width
                for col in columns:
                    width[col] = len(name[col]) + 2

                for varname, var in sorted(
                    mat.variants.items(), key=lambda item: item[0].lower()
                ):  # iterate variants sorted by variant name

                    # Initialize row
                    row = {}

                    # Variant name
                    row["variant"] = "``'" + varname + "'``"
                    if varname == mat.default:
                        row["variant"] += " (default)"

                    # Load medium
                    medium = var.medium

                    if isinstance(medium, Medium2D):
                        row["model"] = ":class:`Medium2D`"
                    elif isinstance(medium, PoleResidue):
                        # Pole number
                        row["model"] = str(len(medium.poles)) + "-pole"
                        # Lossy (based on model)
                        nonzero = np.sum(np.abs(np.array(medium.poles).real))
                        if nonzero:
                            if medium.loss_upper_bound < LOW_LOSS_THRESHOLD:
                                row["model"] += ", low loss"
                            else:
                                row["model"] += ", lossy"
                        else:
                            row["model"] += ", lossless"

                    else:
                        row["model"] = ""

                    # Valid range
                    if medium.frequency_range is None:
                        row["range"] = "Not specified"
                    else:
                        freq = np.array(medium.frequency_range) / 1e12  # THz
                        if freq[0] == 0:
                            wl = np.array([np.inf, C_0 / (freq[1] * 1e12)])  # um
                        else:
                            wl = C_0 / (freq * 1e12)  # um
                        ev = 1.2398 / wl  # eV
                        unit_disp = unit
                        if unit == "um":
                            rng = wl
                            # unit_disp = '$\mu$m' # for .md file
                            unit_disp = r":math:`{\mu}m`"
                        elif unit == "THz":
                            rng = freq
                        elif unit == "eV":
                            rng = ev
                        row["range"] = (
                            num2str(min(rng)) + " - " + num2str(max(rng)) + " " + unit_disp
                        )

                    # Reference
                    if var.reference is not None:
                        row["ref"] = ""
                        for ref in var.reference:
                            if ref in ref_list:
                                row["ref"] += "[" + num2str(ref_list.index(ref) + 1) + "]"
                            else:
                                row["ref"] += "[" + num2str(len(ref_list) + 1) + "]"
                                ref_list.append(ref)

                    # Data
                    if var.data_url is not None:
                        row["ref"] += " `[data] <" + var.data_url + ">`__"

                    # Update table contents
                    rows.append(row)

                    # Update table column width
                    for col in columns:
                        if width[col] < len(row[col]):
                            width[col] = len(row[col])

                    # Update example code
                    code_string += (
                        ">>> medium = material_library['" + abbr + "']['" + varname + "']\n\n"
                    )

                # Write table
                tab = "   "
                divider = tab + " ".join("=" * width[col] for col in columns) + "\n"
                f.write(".. table::\n")
                f.write(tab + ":widths: auto\n\n")
                f.write(divider)
                f.write(
                    tab
                    + " ".join(name[col] + " " * (width[col] - len(name[col])) for col in columns)
                    + "\n"
                )
                f.write(divider)
                for row in rows:
                    f.write(
                        tab
                        + " ".join(row[col] + " " * (width[col] - len(row[col])) for col in columns)
                        + "\n"
                    )
                f.write(divider)
                f.write("\n")

                # Write code example
                if len(code_string) != 0:
                    f.write("Examples:\n\n")
                    f.write(code_string)

                # Extract reference from database
                ref_string = ""
                for ref in ref_list:
                    if ref.journal is not None:
                        ref_string += "#. \\" + ref.journal
                    for link in ["doi", "url"]:
                        value = getattr(ref, link)
                        if value is not None:
                            ref_string += " `[" + link + "] <" + value + ">`__"
                    ref_string += "\n"

                # Write references
                if len(ref_string) != 0:
                    f.write("References:\n\n")
                    f.write(ref_string)
                    f.write("\n")


def main():
    generate_material_library_doc()


if __name__ == "__main__":
    main()
