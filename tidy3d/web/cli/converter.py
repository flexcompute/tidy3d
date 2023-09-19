from pathlib import Path


def _remove_spaces(line: str) -> str:
    """
    Removes spaces from line, except spaces found between quotes. Also converts " into ' for ease of further parsing
    """
    quotes = False
    comment = False
    new_line = ""
    for c in line:
        if c == " " and not quotes:
            if comment:
                new_line += c
        else:
            if c == "'" or c == '"':
                quotes = not quotes
                c = "'"
            elif c == "#":
                comment = True
            new_line += c
    return new_line


def _is_int(number_str: str) -> bool:
    try:
        int(number_str)
        return True
    except ValueError:
        return False


def _is_float(number_str: str) -> bool:
    try:
        float(number_str)
        return True
    except ValueError:
        return False


def _clean(sci_str: str) -> str:
    number, power = sci_str.split("e")
    while number[-1] == "0" or number[-1] == ".":
        if len(number) == 1:
            break
        number = number[:-1]
    while power[1] == "0" and len(power) > 2:
        power = power[0] + power[2:]
    if power[0] == "+":
        power = power[1:]
    if power == "0":
        return number
    if "." not in number and power == "1":
        return number + "0"
    return number + "e" + power


def _to_um(array: list) -> list:
    output_array = []
    for number_str in array:
        if number_str == "None":
            output_array.append("None")
        elif number_str == "td.inf" or number_str == "-td.inf":
            output_array.append(number_str)
        elif "e" in number_str:  # potentially scientific notation
            number_power = number_str.split("e")
            if len(number_power) != 2:  # number_str is a variable name with an 'e'
                output_array.append(number_str + "*1e6")
            else:
                number, power = number_power
                if _is_int(number):
                    if _is_int(power):
                        value = int(number) * 10 ** (int(power)) * 1e6
                        value_str = _clean(f"{value:e}")
                        output_array.append(value_str)
                    elif _is_float(power):
                        value = int(number) * 10 ** (float(power)) * 1e6
                        value_str = _clean(f"{value:e}")
                        output_array.append(value_str)
                    else:  # number_str something else
                        output_array.append(number_str + "*1e6")
                elif _is_float(number):
                    if _is_int(power):
                        value = float(number) * 10 ** (int(power)) * 1e6
                        value_str = _clean(f"{value:e}")
                        output_array.append(value_str)
                    elif _is_float(power):
                        value = float(number) * 10 ** (float(power)) * 1e6
                        value_str = _clean(f"{value:e}")
                        output_array.append(value_str)
                    else:  # number_str is something else
                        output_array.append(number_str + "*1e6")
                else:
                    output_array.append(number_str + "*1e6")
        elif _is_int(number_str):
            value_str = _clean(f"{int(number_str):e}")
            output_array.append(value_str)
        elif _is_float(number_str):
            value_str = _clean(f"{float(number_str):e}")
            output_array.append(value_str)
        else:  # dealing with a string representing a variable
            output_array.append(number_str + "*1e6")
    return output_array


def _is_declaration(Line: str) -> bool:
    comment = False
    for i in range(0, len(Line)):
        if Line[i] == "#":
            comment = True
        if Line[i] == "=" and not comment:
            if "[" in Line[i] and "]" in Line[i]:  # change array syntax
                l_bracket = Line[i].find("[")
                new_array_string = "["
                array = Line[l_bracket + 1 :].split(";")
                for a in array:
                    new_array_string += "(" + a + "),"
                r_bracket = new_array_string.find("]")
                new_array_string = (
                    new_array_string[:r_bracket] + ")" + new_array_string[r_bracket:-1]
                )
                Line[i] = Line[i][:l_bracket] + new_array_string
            return True
    return False


def _add_to_variable_dict(line: str, variable_dict: dict):
    """
    Takes a string which describes a variable assignment and adds the variable assignment to the variable dictionary. Supported for strings, floats, booleans, and other variables.

    E.g. "string_var='str'" adds 'string_var': 'str' to the dictionary
    """
    variable_name = ""
    var = False
    for i in range(0, len(line)):
        if line[i] != "=" and not var:
            variable_name += line[i]
        else:
            if line[i] == "=":
                var = True
            else:
                if line[i] == "'":
                    # if variable represents a string
                    variable_dict[variable_name] = line[i + 1 :].split("'")[0]
                elif line[i : i + 4] == "True":
                    variable_dict[variable_name] = True
                elif line[i : i + 5] == "False":
                    variable_dict[variable_name] = False
                elif line[i] == "[":
                    return  # is an array
                else:
                    try:  # if variable represents a float
                        float(line[i:].split(";")[0])
                        variable_dict[variable_name] = float(line[i:].split(";")[0])
                    except ValueError:
                        # if variable represents another variable
                        variable_dict[variable_name] = variable_dict[line[i:].split(";")[0]]
                return


def _addrect(Lines, i: int, rect: int) -> str:
    # https://optics.ansys.com/hc/en-us/articles/360034901473-Rectangle-Simulation-Object
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    x_min, x_max = "0", "0"
    y_min, y_max = "0", "0"
    z_min, z_max = "0", "0"
    name = "rect_" + str(rect)
    medium = ""
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "x_min":
            x_min = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "x_max":
            x_max = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "y_min":
            y_min = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "y_max":
            y_max = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "z_min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:10] == "z_max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:10] == "index":
            index = Lines[i][12:].split(";")[0][:-1]
            medium = "td.Medium(permittivity=" + index + "**2)"
        elif Lines[i][5:13] == "material":
            medium = Lines[i][15:].split(";")[0][:-1] + " # NOTE: check material library"
        i += 1
        if i == len(Lines):
            break
    rect_string = name.replace(" ", "_") + " = td.Structure(\n"
    rect_string += "\tgeometry=td.Box(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    rect_string += "\t\tcenter=(" + x + "," + y + "," + z + "),\n"
    rect_string += "\t\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    rect_string += "\t),\n"
    rect_string += "\tmedium=" + medium + ",\n"
    rect_string += "\tname='" + name + "',\n"
    rect_string += ")\n"
    return rect_string, name.replace(" ", "_")


def _addsphere(Lines, i: int, sph: int):
    # https://optics.ansys.com/hc/en-us/articles/360034901553-Sphere-Simulation-Object
    x, y, z = "0", "0", "0"
    radius = " # NOTE: Please specify"
    name = "sphere_" + str(sph)
    medium = "td.Medium()"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "radius":
            radius = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:10] == "index":
            index = Lines[i][12:].split(";")[0][:-1]
            medium = "td.Medium(permittivity=" + index + "**2)"
        elif Lines[i][5:13] == "material":
            medium = Lines[i][15:].split(";")[0][:-1] + " # NOTE: check material library"
        elif Lines[i][5:19] == "make ellipsoid":
            pass  # for now
        i += 1
        if i == len(Lines):
            break
    sphere_string = name.replace(" ", "_") + " = td.Structure(\n"
    sphere_string += "\tgeometry=td.Sphere(\n"
    x, y, z, radius = _to_um([x, y, z, radius])
    sphere_string += "\t\tcenter=(" + x + "," + y + "," + z + "),\n"
    sphere_string += "\t\tradius=" + radius + ",\n"
    sphere_string += "\t),\n"
    sphere_string += "\tmedium=" + medium + ",\n"
    sphere_string += "\tname='" + name + "',\n"
    sphere_string += ")\n"
    return sphere_string, name.replace(" ", "_")


def _addcircle(Lines, i: int, cyl: int):
    # https://optics.ansys.com/hc/en-us/articles/360034901513-Circle-Simulation-Object
    x, y, z = "0", "0", "0"
    z_min, z_max = "-td.inf/2", "td.inf/2"
    z_span = z_max + "-" + z_min
    name = "cylinder_" + str(cyl)
    radius = " # NOTE: please specify"
    medium = "td.Medium()"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "z min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            z_span = z_max + "-" + z_min
        elif Lines[i][5:10] == "z max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            z_span = z_max + "-" + z_min
        elif Lines[i][5:11] == "z span":
            z_span = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "radius":
            radius = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:10] == "index":
            index = Lines[i][12:].split(";")[0][:-1]
            medium = "td.Medium(permittivity=" + index + "**2)"
        elif Lines[i][5:13] == "material":
            medium = Lines[i][15:].split(";")[0][:-1] + " # NOTE: check material library"
        i += 1
        if i == len(Lines):
            break
    cylinder_string = name.replace(" ", "_") + " = td.Structure(\n"
    cylinder_string += "\tgeometry=td.Cylinder(\n"
    x, y, z, radius, z_span = _to_um([x, y, z, radius, z_span])
    cylinder_string += "\t\tcenter=(" + x + "," + y + "," + z + "),\n"
    cylinder_string += "\t\tradius=" + radius + ",\n"
    cylinder_string += "\t\tlength=" + z_span + ",\n"
    cylinder_string += "\t),\n"
    cylinder_string += "\tmedium=" + medium + ",\n"
    cylinder_string += "\tname='" + name + "'\n"
    cylinder_string += ")\n"
    return cylinder_string, name.replace(" ", "_")


def _addpoly(Lines, i: int, polyslab: int):
    vertices = "[ # NOTE: please specify ]"
    z_min, z_max = "0", "0"
    z_span = "z_min - z_max"
    axis = "2"
    name = "polyslab_" + str(polyslab)
    medium = "td.Medium()"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:10] == "z min":
            z_min = Lines[i][12:].split(";")[0][:-1]
        elif Lines[i][5:10] == "z max":
            z_max = Lines[i][12:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            z_span = Lines[i][13:].split(";")[0][:-1]
            z_min, z_max = "-" + z_span + "/2", z_span + "/2"
        elif Lines[i][5:10] == "index":
            index = Lines[i][12:].split(";")[0][:-1]
            medium = "td.Medium(permittivity=" + index + "**2)"
        elif Lines[i][5:13] == "material":
            medium = Lines[i][15:].split(";")[0][:-1] + " # NOTE: check material library"
        elif Lines[i][5:13] == "vertices":
            vertices = Lines[i][15:].split(";")[0][:-1]
        i += 1
        if i == len(Lines):
            break
    poly_string = name.replace(" ", "_") + " = td.Structure(\n"
    poly_string += "\tgeometry=td.PolySlab(\n"
    poly_string += "\t\tvertices=" + vertices + ",\n"
    poly_string += "\t\taxis=" + axis + ",\n"
    z_min, z_max = _to_um([z_min, z_max])
    poly_string += "\t\tslab_bounds=(" + z_min + "," + z_max + "),\n"
    poly_string += "\t),\n"
    poly_string += "\tmedium=" + medium + ",\n"
    poly_string += "\tname='" + name + "'\n"
    poly_string += ")\n"
    return poly_string, name.replace(" ", "_")


def _adddipole(Lines, i: int, ptdipole: int):
    # https://optics.ansys.com/hc/en-us/articles/360034382854-Plane-wave-and-beam-source-Simulation-object
    x, y, z = "0", "0", "0"
    name = "point_dipole_" + str(ptdipole)
    polarization = "Ez"
    # ---------------- source parameters ----------------
    source_name = "dipole_source_time_" + str(ptdipole)
    wavelength_start = "0.4"
    wavelength_stop = "0.7"
    freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
    fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
    offset = "5.0"
    # ---------------------------------------------------
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:21] == "wavelength start":
            wavelength_start = "(" + Lines[i][23:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:20] == "wavelength stop":
            wavelength_stop = "(" + Lines[i][22:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "frequency":
            freq0 = Lines[i][16:].split(";")[0][:-1] + "*1e12"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "bandwidth":
            fwidth = Lines[i][16:].split(";")[0][:-1] + "*1e12"
        elif Lines[i][5:11] == "offset":
            offset = Lines[i][13:].split(";")[0][:-1] + "*1e-15"
        i += 1
        if i == len(Lines):
            break
    dipole_string = source_name + " = td.GaussianPulse(\n"
    dipole_string += "\tfreq0=" + freq0 + ",\n"
    dipole_string += "\tfwidth=" + fwidth + ",\n"
    dipole_string += "\toffset=" + offset + "\n"
    dipole_string += ")\n"
    dipole_string += name.replace(" ", "_") + " = td.PointDipole(\n"
    x, y, z = _to_um([x, y, z])
    dipole_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    dipole_string += "\tsource_time=" + source_name + ",\n"
    dipole_string += "\tname='" + name + "',\n"
    dipole_string += "\tpolarization='" + polarization + "',\n"
    dipole_string += ")\n"
    return dipole_string, name.replace(" ", "_")


def _addgaussian(Lines, i: int, gauss: int):
    # https://optics.ansys.com/hc/en-us/articles/360034382854-Plane-wave-and-beam-source-Simulation-object
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "gaussian_beam_" + str(gauss)
    direction = "'+'"
    angle_theta = "0"
    angle_phi = "0"
    pol_angle = "0"
    waist_radius = "1.0"
    waist_distance = "0"
    # ---------------- source parameters ----------------
    source_name = "gauss_source_time_" + str(gauss)
    wavelength_start = "0.4"
    wavelength_stop = "0.7"
    freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
    fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
    offset = "5.0"
    # ---------------------------------------------------
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:14] == "direction":
            if Lines[i][16:].split("'")[0] == "backward":
                direction = "'-'"
        elif Lines[i][5:16] == "angle theta":
            angle_theta = Lines[i][18:].split(";")[0][:-1]
        elif Lines[i][5:14] == "angle phi":
            angle_phi = Lines[i][16:].split(";")[0][:-1]
        elif Lines[i][5:22] == "polarization angle":
            pol_angle = Lines[i][24:].split(";")[0][:-1]
        elif Lines[i][5:20] == "waist radius w0":
            waist_radius = Lines[i][22:].split(";")[0][:-1]
        elif Lines[i][5:24] == "distance from waist":
            waist_distance = Lines[i][26:].split(";")[0][:-1]
        elif Lines[i][5:21] == "wavelength start":
            wavelength_start = "(" + Lines[i][23:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:20] == "wavelength stop":
            wavelength_stop = "(" + Lines[i][22:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "frequency":
            freq0 = Lines[i][16:].split(";")[0][:-1] + "e12"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "bandwidth":
            fwidth = Lines[i][16:].split(";")[0][:-1] + "e12"
        elif Lines[i][5:11] == "offset":
            offset = Lines[i][13:].split(";")[0][:-1] + "e-15"
        i += 1
        if i == len(Lines):
            break
    gauss_string = source_name + " = td.GaussianPulse(\n"
    gauss_string += "\tfreq0=" + freq0 + ",\n"
    gauss_string += "\tfwidth=" + fwidth + ",\n"
    gauss_string += "\toffset=" + offset + "\n"
    gauss_string += ")\n"
    gauss_string += name.replace(" ", "_") + " = td.GaussianBeam(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    gauss_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    gauss_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    gauss_string += "\tsource_time=" + source_name + ",\n"
    gauss_string += "\tname='" + name + "',\n"
    gauss_string += "\tdirection=" + direction + ",\n"
    gauss_string += "\tangle_theta=" + angle_theta + ",\n"
    gauss_string += "\tangle_phi=" + angle_phi + ",\n"
    gauss_string += "\tpol_angle=" + pol_angle + ",\n"
    gauss_string += "\twaist_radius=" + waist_radius + "*1e6,\n"
    gauss_string += "\twaist_distance=" + waist_distance + "*1e6,\n"
    gauss_string += ")\n"
    return gauss_string, name.replace(" ", "_")


def _addplane(Lines, i: int, planewv: int):
    # https://optics.ansys.com/hc/en-us/articles/360034382854-Plane-wave-and-beam-source-Simulation-object
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "plane_source_" + str(planewv)
    direction = "'+'"
    angle_theta = "0"
    angle_phi = "0"
    pol_angle = "0"
    # ---------------- source parameters ----------------
    source_name = "plane_source_time_" + str(planewv)
    wavelength_start = "0.4"
    wavelength_stop = "0.7"
    freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
    fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
    offset = "5.0"
    # ---------------------------------------------------
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:14] == "direction":
            if Lines[i][16:].split("'")[0] == "backward":
                direction = "'-'"
        elif Lines[i][5:16] == "angle theta":
            angle_theta = Lines[i][18:].split(";")[0][:-1]
        elif Lines[i][5:14] == "angle phi":
            angle_phi = Lines[i][16:].split(";")[0][:-1]
        elif Lines[i][5:22] == "polarization angle":
            pol_angle = Lines[i][24:].split(";")[0][:-1]
        elif Lines[i][5:21] == "wavelength start":
            wavelength_start = "(" + Lines[i][23:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:20] == "wavelength stop":
            wavelength_stop = "(" + Lines[i][22:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "frequency":
            freq0 = Lines[i][16:].split(";")[0][:-1] + "e12"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "bandwidth":
            fwidth = Lines[i][16:].split(";")[0][:-1] + "e12"
        elif Lines[i][5:11] == "offset":
            offset = Lines[i][13:].split(";")[0][:-1] + "e-15"
        i += 1
        if i == len(Lines):
            break
    plane_string = source_name + " = td.GaussianPulse(\n"
    plane_string += "\tfreq0=" + freq0 + ",\n"
    plane_string += "\tfwidth=" + fwidth + ",\n"
    plane_string += "\toffset=" + offset + "\n"
    plane_string += ")\n"
    plane_string += name.replace(" ", "_") + " = td.PlaneWave(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    plane_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    plane_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    plane_string += "\tsource_time=" + source_name + ",\n"
    plane_string += "\tname='" + name + "',\n"
    plane_string += "\tdirection=" + direction + ",\n"
    plane_string += "\tangle_theta=" + angle_theta + ",\n"
    plane_string += "\tangle_phi=" + angle_phi + ",\n"
    plane_string += "\tpol_angle=" + pol_angle + ",\n"
    plane_string += ")\n"
    return plane_string, name.replace(" ", "_")


def _addtfsf(Lines, i: int, tfsf: int):
    # https://optics.ansys.com/hc/en-us/articles/360034382854-Plane-wave-and-beam-source-Simulation-object
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "tfsf_" + str(tfsf)
    direction = "'+'"
    angle_theta = "0"
    angle_phi = "0"
    pol_angle = "0"
    # ---------------- source parameters ----------------
    source_name = "tfsf_source_time_" + str(tfsf)
    wavelength_start = "0.4"
    wavelength_stop = "0.7"
    freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
    fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
    offset = "5.0"
    # ---------------------------------------------------
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:14] == "direction":
            if Lines[i][16:].split("'")[0] == "backward":
                direction = "'-'"
        elif Lines[i][5:16] == "angle theta":
            angle_theta = Lines[i][18:].split(";")[0][:-1]
        elif Lines[i][5:14] == "angle phi":
            angle_phi = Lines[i][16:].split(";")[0][:-1]
        elif Lines[i][5:22] == "polarization angle":
            pol_angle = Lines[i][24:].split(";")[0][:-1]
        elif Lines[i][5:20] == "waist radius w0":
            waist_radius = Lines[i][22:].split(";")[0][:-1]
        elif Lines[i][5:24] == "distance from waist":
            waist_distance = Lines[i][26:].split(";")[0][:-1]
        elif Lines[i][5:21] == "wavelength start":
            wavelength_start = "(" + Lines[i][23:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:20] == "wavelength stop":
            wavelength_stop = "(" + Lines[i][22:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "frequency":
            freq0 = Lines[i][16:].split(";")[0][:-1] + "e12"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "bandwidth":
            fwidth = Lines[i][16:].split(";")[0][:-1] + "e12"
        elif Lines[i][5:11] == "offset":
            offset = Lines[i][13:].split(";")[0][:-1] + "e-15"
        i += 1
        if i == len(Lines):
            break
    tfsf_string = source_name + " = td.GaussianPulse(\n"
    tfsf_string += "\tfreq0=" + freq0 + ",\n"
    tfsf_string += "\tfwidth=" + fwidth + ",\n"
    tfsf_string += "\toffset=" + offset + "\n"
    tfsf_string += ")\n"
    tfsf_string += name.replace(" ", "_") + " = td.TFSF(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    tfsf_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    tfsf_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    tfsf_string += "\tsource_time=" + source_name + ",\n"
    tfsf_string += "\tname='" + name + "',\n"
    tfsf_string += "\tdirection=" + direction + ",\n"
    tfsf_string += "\tangle_theta=" + angle_theta + ",\n"
    tfsf_string += "\tangle_phi=" + angle_phi + ",\n"
    tfsf_string += "\tpol_angle=" + pol_angle + ",\n"
    tfsf_string += "\twaist_radius=" + waist_radius + "*1e6,\n"
    tfsf_string += "\twaist_distance=" + waist_distance + "*1e6,\n"
    tfsf_string += ")\n"
    return tfsf_string, name.replace(" ", "_")


def _addmode(Lines, i: int, modesrc: int):
    # https://optics.ansys.com/hc/en-us/articles/360034902153-Mode-source-Simulation-object
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "mode_source_" + str(modesrc)
    direction = "'+'"
    num_modes = "1"
    angle_theta = "0"
    mode_spec_name = "mode_spec_" + str(modesrc)
    mode_index = "0"
    # ---------------- source parameters ----------------
    source_name = "mode_source_time_" + str(modesrc)
    wavelength_start = "0.4"
    wavelength_stop = "0.7"
    freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
    fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
    offset = "5.0"
    # ---------------------------------------------------
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:13] == "direction":
            if Lines[i][16:].split("'")[0] == "Backward":
                direction = "'-'"
        elif Lines[i][5:23] == "polarization angle":
            angle_theta = Lines[i][25:].split(";")[0][:-1]
        elif Lines[i][5:19] == "mode selection":
            if Lines[i][22:].split(";")[0][:-1] == "'fundamental TM mode'":
                num_modes, mode_index = "2", "1"
            elif Lines[i][22:].split("'")[0] == "user select":
                num_modes, mode_index = (
                    "PLEASE SPECIFY NUMBER OF MODES",
                    "PLEASE SPECIFY MODE INDEX",
                )
        elif Lines[i][5:21] == "wavelength start":
            wavelength_start = "(" + Lines[i][23:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:20] == "wavelength stop":
            wavelength_stop = "(" + Lines[i][22:].split(";")[0][:-1] + "*1e6)"
            freq0 = "(td.C_0/" + wavelength_start + " + td.C_0/" + wavelength_stop + ")/2"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "frequency":
            freq0 = Lines[i][16:].split(";")[0][:-1] + "e12"
            if wavelength_start == wavelength_stop:
                fwidth = freq0 + "/10"
            else:
                fwidth = "(td.C_0/" + wavelength_start + " - td.C_0/" + wavelength_stop + ")"
        elif Lines[i][5:14] == "bandwidth":
            fwidth = Lines[i][16:].split(";")[0][:-1] + "e12"
        elif Lines[i][5:11] == "offset":
            offset = Lines[i][13:].split(";")[0][:-1] + "e-15"
        i += 1
        if i == len(Lines):
            break
    mode_string = source_name + " = td.GaussianPulse(\n"
    mode_string += "\tfreq0=" + freq0 + ",\n"
    mode_string += "\tfwidth=" + fwidth + ",\n"
    mode_string += "\toffset=" + offset + "\n"
    mode_string += ")\n"
    mode_string += mode_spec_name + " = td.ModeSpec(num_modes=" + num_modes
    mode_string += ",angle_theta=" + angle_theta + ")\n"
    mode_string += name.replace(" ", "_") + " = td.ModeSource(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    mode_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    mode_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    mode_string += "\tsource_time=" + source_name + ",\n"
    mode_string += "\tname='" + name + "',\n"
    mode_string += "\tmode_spec=" + mode_spec_name + ",\n"
    if mode_index != -1:
        mode_string += "\tmode_index=" + mode_index + ",\n"
    mode_string += "\tdirection=" + direction + ",\n"
    mode_string += ")\n"
    return mode_string, name.replace(" ", "_")


def _get_BCs(dim_bc_file: str):
    dim_bc, dim_sym = "td.PML(num_layers=", "0"
    if dim_bc_file == "Periodic":
        dim_bc = "td.Periodic()"
    elif dim_bc_file == "Metal":
        dim_bc = "td.PECBoundary()"
    elif dim_bc_file == "Symmetric":
        dim_sym = "1"
    elif dim_bc_file == "Anti-Symmetric":
        dim_sym = "-1"
    elif dim_bc_file == "PMC":
        dim_bc = "td.PMCBoundary()"
    elif dim_bc_file == "Bloch":
        dim_bc = "td.BlochBoundary(bloch_vec=1)"
    return dim_bc, dim_sym


def _addfdtd(Lines, i: int, v_dict: dict) -> str:
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    x_min, x_max = "0", "0"
    y_min, y_max = "0", "0"
    z_min, z_max = "0", "0"
    background_index = "1"
    runtime = " # NOTE: please specify"
    mesh_accuracy = 1
    min_steps_per_wvl = 6
    num_layers = "12"
    x_min_bc = "td.PML(num_layers="
    x_max_bc = x_min_bc
    y_min_bc, y_max_bc = x_min_bc, x_min_bc
    z_min_bc, z_max_bc = x_min_bc, x_min_bc
    x_sym, y_sym, z_sym = "0", "0", "0"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "x_min":
            x_min = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "x_max":
            x_max = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "y_min":
            y_min = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "y_max":
            y_max = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "z_min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:10] == "z_max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:21] == "background index":
            background_index = Lines[i][23:].split(";")[0][:-1]
        elif Lines[i][5:18] == "mesh accuracy":
            mesh_accuracy = v_dict[Lines[i][20:].split(";")[0][:-1]]
            min_steps_per_wvl = str(int(2 + 4 * mesh_accuracy))
        elif Lines[i][5:13] == "x min bc":  # min
            x_min_bc_file = Lines[i][16:].split("'")[0]
            x_min_bc, x_sym = _get_BCs(x_min_bc_file)
        elif Lines[i][5:13] == "y min bc":
            y_min_bc_file = Lines[i][16:].split("'")[0]
            y_min_bc, y_sym = _get_BCs(y_min_bc_file)
        elif Lines[i][5:13] == "z min bc":
            z_min_bc_file = Lines[i][16:].split("'")[0]
            z_min_bc, z_sym = _get_BCs(z_min_bc_file)
        elif Lines[i][5:13] == "x max bc":  # max
            x_max_bc_file = Lines[i][16:].split("'")[0]
            x_max_bc, x_sym = _get_BCs(x_max_bc_file)
        elif Lines[i][5:13] == "y max bc":
            y_max_bc_file = Lines[i][16:].split("'")[0]
            y_max_bc, y_sym = _get_BCs(y_max_bc_file)
        elif Lines[i][5:13] == "z max bc":
            z_max_bc_file = Lines[i][16:].split("'")[0]
            z_max_bc, z_sym = _get_BCs(z_max_bc_file)
        elif Lines[i][5:15] == "pml layers":
            num_layers = Lines[i][17:].split(";")[0][:-1]
        elif Lines[i][5:20] == "simulation time":
            runtime = Lines[i][22:].split(";")[0][:-1]
        i += 1
        if i == len(Lines):
            break
    if xl in v_dict:
        if v_dict[xl] == 0:
            x_min_bc, x_max_bc = "td.Periodic()", "td.Periodic()"
    elif xl == "0":
        x_min_bc, x_max_bc = "td.Periodic()", "td.Periodic()"
    if yl in v_dict:
        if v_dict[yl] == 0:
            y_min_bc, y_max_bc = "td.Periodic()", "td.Periodic()"
    elif yl == "0":
        y_min_bc, y_max_bc = "td.Periodic()", "td.Periodic()"
    if zl in v_dict:
        if v_dict[zl] == 0:
            z_min_bc, z_max_bc = "td.Periodic()", "td.Periodic()"
    elif zl == "0":
        z_min_bc, z_max_bc = "td.Periodic()", "td.Periodic()"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    sim_string = "sim = td.Simulation(\n"
    sim_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    sim_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    sim_string += "\trun_time=" + runtime + ",\n"
    sim_string += "\tmedium=td.Medium(permittivity=" + background_index + "**2),\n"
    sim_string += "\tsymmetry=(" + x_sym + "," + y_sym + "," + z_sym + "),\n"
    sim_string += "\tboundary_spec=td.BoundarySpec(\n"
    sim_string += "\t\tx = td.Boundary(minus=" + x_min_bc
    if x_min_bc[-1] == "=":  # it's a pml boundary
        sim_string += num_layers + ")"
    sim_string += ", plus=" + x_max_bc
    if x_max_bc[-1] == "=":  # ti's a pml boundary
        sim_string += num_layers + ")"
    sim_string += "),\n"
    sim_string += "\t\ty = td.Boundary(minus=" + y_min_bc
    if y_min_bc[-1] == "=":  # it's a pml boundary
        sim_string += num_layers + ")"
    sim_string += ", plus=" + y_max_bc
    if y_max_bc[-1] == "=":  # ti's a pml boundary
        sim_string += num_layers + ")"
    sim_string += "),\n"
    sim_string += "\t\tz = td.Boundary(minus=" + z_min_bc
    if z_min_bc[-1] == "=":  # it's a pml boundary
        sim_string += num_layers + ")"
    sim_string += ", plus=" + z_max_bc
    if z_max_bc[-1] == "=":  # ti's a pml boundary
        sim_string += num_layers + ")"
    sim_string += "),\n\t),\n"
    sim_string += "\t"
    sim_string += "grid_spec="
    gridspec_string = "td.GridSpec.auto(min_steps_per_wvl=" + min_steps_per_wvl + ","
    gridspec_string += " wavelength=1.0)"
    sim_string += gridspec_string
    sim_string += ",\n)\n"
    return sim_string, gridspec_string


def _addmesh(Lines, i: int, overstrct: int):
    x, y, z = "0", "0", "0"
    xl, yl, zl = "td.inf", "td.inf", "td.inf"
    dx, dy, dz = "None", "None", "None"
    name = "refine_box_" + str(overstrct)
    override_string = " = td.MeshOverrideStructure(\n"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:7] == "dx":
            dx = Lines[i][9:].split(";")[0][:-1]
        elif Lines[i][5:7] == "dy":
            dy = Lines[i][9:].split(";")[0][:-1]
        elif Lines[i][5:7] == "dz":
            dz = Lines[i][9:].split(";")[0][:-1]
        i += 1
        if i == len(Lines):
            break
    x, y, z, xl, yl, zl, dx, dy, dz = _to_um([x, y, z, xl, yl, zl, dx, dy, dz])
    override_string += "\tgeometry=td.Box(center=(" + x + "," + y + "," + z + "),"
    override_string += "size=(" + xl + "," + yl + "," + zl + ")),\n"
    override_string += "\tdl=(" + dx + "," + dy + "," + dz + "),\n)\n"
    return override_string, name.replace(" ", "_")


def _addindex(Lines, i: int, indexmon: int):
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "permittivity_monitor_" + str(indexmon)
    freqs = " # NOTE: please specify"
    wave_center, wave_span = "0.5", "0.1"
    index_monitor_string = " = td.PermittivityMonitor(\n"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:22] == "wavelength center":
            wave_center = Lines[i][24:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:20] == "wavelength span":
            wave_span = Lines[i][22:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:21] == "frequency points":
            num_freqs = Lines[i][23:].split(";")[0][:-1]
            freqs = (
                "np.linspace("
                + wave_center
                + "-"
                + wave_span
                + "/2,"
                + wave_center
                + "+"
                + wave_span
                + "/2,"
                + num_freqs
                + ")"
            )
        i += 1
        if i == len(Lines):
            break
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    index_monitor_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    index_monitor_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    index_monitor_string += "\tname='" + name + "',\n"
    index_monitor_string += "\tinterval_space=(1,1,1),\n"
    index_monitor_string += "\tfreqs=" + freqs + ",\n"
    index_monitor_string += ")\n"
    return index_monitor_string, name.replace(" ", "_")


def _addpower(Lines, i: int, fluxmon: int):
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    name = "flux_monitor_" + str(fluxmon)
    freqs = " # NOTE: please specify"
    wave_center, wave_span = "0.5", "0.1"
    flux_monitor_string = " = td.FluxMonitor(\n"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:22] == "wavelength center":
            wave_center = Lines[i][24:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:20] == "wavelength span":
            wave_span = Lines[i][22:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:21] == "frequency points":
            num_freqs = Lines[i][23:].split(";")[0][:-1]
            freqs = (
                "np.linspace("
                + wave_center
                + "-"
                + wave_span
                + "/2,"
                + wave_center
                + "+"
                + wave_span
                + "/2,"
                + num_freqs
                + ")"
            )
        i += 1
        if i == len(Lines):
            break
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    flux_monitor_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    flux_monitor_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    flux_monitor_string += "\tname='" + name + "',\n"
    flux_monitor_string += "\tfreqs=" + freqs + ",\n"
    flux_monitor_string += ")\n"
    return flux_monitor_string, name.replace(" ", "_")


def _addmovie(Lines, i: int, fieldTime: int):
    # https://optics.ansys.com/hc/en-us/articles/360034924633-addefieldmonitor
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    x_min, x_max = "0", "0"
    y_min, y_max = "0", "0"
    z_min, z_max = "0", "0"
    name = "field_time_monitor_" + str(fieldTime)
    field_monitor_string = " = td.FieldTimeMonitor(\n"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "x_min":
            x_min = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "x_max":
            x_max = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "y_min":
            y_min = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "y_max":
            y_max = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "z_min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:10] == "z_max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        i += 1
        if i == len(Lines):
            break
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    field_monitor_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    field_monitor_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    field_monitor_string += "\tname='" + name + "',\n"
    field_monitor_string += "\tinterval=1,\n"
    field_monitor_string += "\tstart=0.0,\n"
    field_monitor_string += "\tstop=None,\n"
    field_monitor_string += ")\n"
    return field_monitor_string, name.replace(" ", "_")


def _addefieldmonitor(Lines, i: int, fieldmon: int):
    # https://optics.ansys.com/hc/en-us/articles/360034924633-addefieldmonitor
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    x_min, x_max = "0", "0"
    y_min, y_max = "0", "0"
    z_min, z_max = "0", "0"
    name = "field_monitor_" + str(fieldmon)
    freqs = " # NOTE: please specify"
    wave_center, wave_span = "0.5", "0.1"
    field_monitor_string = " = td.FieldMonitor(\n"
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "x_min":
            x_min = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "x_max":
            x_max = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "y_min":
            y_min = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "y_max":
            y_max = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "z_min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:10] == "z_max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:22] == "wavelength center":
            wave_center = Lines[i][24:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:20] == "wavelength span":
            wave_span = Lines[i][22:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:21] == "frequency points":
            num_freqs = Lines[i][23:].split(";")[0][:-1]
            freqs = (
                "np.linspace("
                + wave_center
                + "-"
                + wave_span
                + "/2,"
                + wave_center
                + "+"
                + wave_span
                + "/2,"
                + num_freqs
                + ")"
            )
        # add monitor type if I ever find mapping between the numbers?
        i += 1
        if i == len(Lines):
            break
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    field_monitor_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    field_monitor_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    field_monitor_string += "\tname='" + name + "',\n"
    field_monitor_string += "\tfreqs=" + freqs + ",\n"
    field_monitor_string += ")\n"
    return field_monitor_string, name.replace(" ", "_")


def _addmodeexpansion(Lines, i: int, modemon: int):
    # https://optics.ansys.com/hc/en-us/articles/360034902413
    x, y, z = "0", "0", "0"
    xl, yl, zl = "0", "0", "0"
    x_min, x_max = "0", "0"
    y_min, y_max = "0", "0"
    z_min, z_max = "0", "0"
    name = "mode_monitor_" + str(modemon)
    num_modes = "1"
    mode = "0"
    freqs = " # NOTE: please specify"
    wave_center, wave_span = "0.5", "0.1"
    mode_spec_name = "mode_spec_" + str(modemon)
    mode_monitor_string = mode_spec_name + " = td.ModeSpec(num_modes="
    while Lines[i][0] == "\n" or Lines[i][:5] == "set('" or Lines[i][0] == "#":
        if Lines[i][5:9] == "name":
            name = Lines[i][12:].split("'")[0]
        elif Lines[i][5:7] == "x'":
            x = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "y'":
            y = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:7] == "z'":
            z = Lines[i][8:].split(";")[0][:-1]
        elif Lines[i][5:10] == "x_min":
            x_min = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "x_max":
            x_max = Lines[i][12:].split(";")[0][:-1]
            xl = x_max + "-" + x_min
        elif Lines[i][5:10] == "y_min":
            y_min = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "y_max":
            y_max = Lines[i][12:].split(";")[0][:-1]
            yl = y_max + "-" + y_min
        elif Lines[i][5:10] == "z_min":
            z_min = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:10] == "z_max":
            z_max = Lines[i][12:].split(";")[0][:-1]
            zl = z_max + "-" + z_min
        elif Lines[i][5:11] == "x span":
            xl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "y span":
            yl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:11] == "z span":
            zl = Lines[i][13:].split(";")[0][:-1]
        elif Lines[i][5:19] == "mode selection":
            mode = Lines[i][22:].split(";")[0][:-1]
            if mode == "'fundamental TM mode'":
                mode = "1"
                num_modes = "2"
            else:
                mode = "0"
        elif Lines[i][5:22] == "wavelength center":
            wave_center = Lines[i][24:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:20] == "wavelength span":
            wave_span = Lines[i][22:].split(";")[0][:-1] + "*1e6"
        elif Lines[i][5:21] == "frequency points":
            num_freqs = Lines[i][23:].split(";")[0][:-1]
            freqs = (
                "np.linspace("
                + wave_center
                + "-"
                + wave_span
                + "/2,"
                + wave_center
                + "+"
                + wave_span
                + "/2,"
                + num_freqs
                + ")"
            )
        i += 1
        if i == len(Lines):
            break
    mode_monitor_string += num_modes + ") # NOTE: check if solving for correct mode!\n"
    mode_monitor_string += name.replace(" ", "_") + " = td.ModeMonitor(\n"
    x, y, z, xl, yl, zl = _to_um([x, y, z, xl, yl, zl])
    mode_monitor_string += "\tcenter=(" + x + "," + y + "," + z + "),\n"
    mode_monitor_string += "\tsize=(" + xl + "," + yl + "," + zl + "),\n"
    mode_monitor_string += "\tfreqs=" + freqs + ",\n"
    mode_monitor_string += "\tmode_spec=" + mode_spec_name + ",\n"
    mode_monitor_string += "\tname='" + name + "',\n"
    mode_monitor_string += ")\n"
    return mode_monitor_string, name.replace(" ", "_")


def lsf_reader(filename: str) -> None:
    tidy3d_file = "import numpy as np\n"
    tidy3d_file += "import matplotlib.pyplot as plt\n"
    tidy3d_file += "import tidy3d as td\n"
    tidy3d_file += "import tidy3d.web as web\n"
    tidy3d_file += "\n"

    structures = "["
    rect, sph, cyl, polyslab = 0, 0, 0, 0
    sources, modesrc, planewv, ptdipole, gauss, tfsf = "[", 0, 0, 0, 0, 0
    monitors = "["
    modemon, fluxmon, fieldmon, fieldTime, indexmon = 0, 0, 0, 0, 0
    gridspec_string = "grid_spec=td.GridSpec.auto()"
    override_structures, overstrct = "[", 0

    with open(filename) as file_lsf:
        Lines = file_lsf.readlines()
    variable_dict = {}
    for i, line in enumerate(Lines):
        Lines[i] = _remove_spaces(line)
    for i, line in enumerate(Lines):
        line = Lines[i]
        if line[0] == "#":
            tidy3d_file += line
        elif line[0] == "\n":
            tidy3d_file += line
        elif _is_declaration(line):
            _add_to_variable_dict(line, variable_dict)
            tidy3d_file += line.replace(";", " ")
        elif line[:8] == "addrect;":
            addrect_string, rect_name = _addrect(Lines, i + 1, rect)
            tidy3d_file += addrect_string
            structures += rect_name + ", "
            rect += 1
        elif line[:10] == "addsphere;":
            addsphere_string, sphere_name = _addsphere(Lines, i + 1, sph)
            tidy3d_file += addsphere_string
            structures += sphere_name + ", "
            sph += 1
        elif line[:10] == "addcircle;":
            addcircle_string, cylinder_name = _addcircle(Lines, i + 1, cyl)
            tidy3d_file += addcircle_string
            structures += cylinder_name + ", "
            cyl += 1
        elif line[:8] == "addpoly;":
            addpoly_string, poly_name = _addpoly(Lines, i + 1, polyslab)
            tidy3d_file += addpoly_string
            structures += poly_name + ", "
            polyslab += 1
        elif line[:10] == "adddipole;":
            adddipole_string, dipole_name = _adddipole(Lines, i + 1, ptdipole)
            tidy3d_file += adddipole_string
            sources += dipole_name + ", "
            ptdipole += 1
        elif line[:12] == "addgaussian;":
            addgaussian_string, gaussian_name = _addgaussian(Lines, i + 1, gauss)
            tidy3d_file += addgaussian_string
            sources += gaussian_name + ", "
            gauss += 1
        elif line[:9] == "addplane;":
            addplane_string, plane_name = _addplane(Lines, i + 1, planewv)
            tidy3d_file += addplane_string
            sources += plane_name + ", "
            planewv += 1
        elif line[:8] == "addtfsf;":
            addtfsf_string, tfsf_name = _addtfsf(Lines, i + 1, tfsf)
            tidy3d_file += addtfsf_string
            sources += tfsf_name + ", "
            tfsf += 1
        elif line[:8] == "addmode;":
            addmode_string, mode_src_name = _addmode(Lines, i + 1, modesrc)
            tidy3d_file += addmode_string
            sources += mode_src_name + ", "
            modesrc += 1
        elif line[:8] == "addfdtd;":
            addfdtd_string, gridspec_string = _addfdtd(Lines, i + 1, variable_dict)
            tidy3d_file += addfdtd_string
        elif line[:8] == "addmesh;":
            override_string, refine_box_name = _addmesh(Lines, i + 1, overstrct)
            override_structures += refine_box_name + ", "
            tidy3d_file += refine_box_name + override_string
            overstrct += 1
        elif (
            line[:9] == "addindex;"
            or line[:18] == "addeffectiveindex;"
            or line[:12] == "addemeindex;"
        ):
            addindex_string, index_name = _addindex(Lines, i + 1, indexmon)
            monitors += index_name + ", "
            tidy3d_file += index_name + addindex_string
            indexmon += 1
        elif line[:9] == "addpower;":
            addpower_string, power_name = _addpower(Lines, i + 1, fluxmon)
            monitors += power_name + ", "
            tidy3d_file += power_name + addpower_string
            fluxmon += 1
        elif (
            line[:9] == "addmovie;"
            or line[:22] == "addemfieldtimemonitor;"
            or line[:8] == "addtime;"
        ):
            addmovie_string, movie_name = _addmovie(Lines, i + 1, fieldTime)
            monitors += movie_name + ", "
            tidy3d_file += movie_name + addmovie_string
            fieldTime += 1
        elif (
            line[:17] == "addefieldmonitor;"
            or line[:11] == "addprofile;"
            or line[:18] == "addemfieldmonitor;"
            or line[:14] == "addemeprofile;"
        ):
            addefieldmonitor_string, monitor_name = _addefieldmonitor(Lines, i + 1, fieldmon)
            monitors += monitor_name + ", "
            tidy3d_file += monitor_name + addefieldmonitor_string
            fieldmon += 1
        elif line[:17] == "addmodeexpansion;":
            addmodeexpansion_string, modemonitor_name = _addmodeexpansion(Lines, i + 1, modemon)
            # not adding name first as we needed to create mode_spec
            tidy3d_file += addmodeexpansion_string
            monitors += modemonitor_name + ", "
            modemon += 1
        elif line[:5] == "set('":
            pass
        elif line[:14] == "switchtolayout":
            pass
        elif line[:9] == "selectall":
            pass
        elif line[:6] == "delete":
            pass
        else:
            tidy3d_file += "# " + line[:-1] + " # NOTE: does not yet parse to Tidy3D\n"
    if len(structures) > 1:
        structures = structures[:-2] + "]"
    else:
        structures = "[]"
    if len(sources) > 1:
        sources = sources[:-2] + "]"
    else:
        sources = "[]"
    if len(monitors) > 1:
        monitors = monitors[:-2] + "]"
    else:
        monitors = "[]"
    if len(override_structures) > 1:
        override_structures = override_structures[:-2] + "]"
    else:
        override_structures = "[]"
    tidy3d_file += "\nsim = sim.copy(update=dict(structures=" + structures + ","
    tidy3d_file += " # NOTE: Check order of structures for potential overlap issues\n"
    tidy3d_file += "\tsources=" + sources + ",\n"
    tidy3d_file += "\tmonitors=" + monitors + ",\n"
    tidy3d_file += (
        "\tgrid_spec="
        + gridspec_string[:-1]
        + ", override_structures="
        + override_structures
        + ")\n\t)\n"
    )
    tidy3d_file += ")"
    return tidy3d_file


def converter_arg(lsf_file, new_file):
    if Path(f"{lsf_file}").suffix != ".lsf":
        raise Exception(f"{lsf_file} must be an .lsf file.")
    if Path(f"{new_file}").suffix != ".py":
        raise Exception(f"{new_file} must be a .py file.")
    file_string = lsf_reader(lsf_file)
    num_specs = file_string.count("NOTE: please specify")
    with open(new_file, "w") as f:
        f.write(file_string)
    print("Created new Tidy3D file:", new_file)
    print(
        f"\nNOTE: This is an experimental feature. Not all scripting commands are supported. Some default variables are specified if not given in the .lsf file, and {num_specs} variables still need to be set in the script in order to run. Check the comments in the output file for hints."
    )
    return
