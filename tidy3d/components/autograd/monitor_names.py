"""Naming contract for adjoint monitors, shared by construction and consumption."""

from __future__ import annotations


def adjoint_monitor_name(index: int, monitor_tag: str) -> str:
    """Adjoint monitor name for one traced structure and data type.

    The shared naming contract between adjoint monitor construction and
    consumption: both sides derive names from the structure index and data-type
    tag alone, so no monitor-name payload is ever stored or transferred.
    """
    monitor_name_map = {
        "fld": f"adjoint_fld_{index}",
        "eps": f"adjoint_eps_{index}",
    }

    # point-cloud monitor families are named systematically per component/side
    if monitor_tag == "fld_pc" or monitor_tag.startswith(("fld_pc_", "eps_pc_")):
        return f"adjoint_{monitor_tag}_{index}"

    if monitor_tag not in monitor_name_map:
        raise KeyError(f"'monitor_tag' must be in {monitor_name_map.keys()}")

    return monitor_name_map[monitor_tag]
