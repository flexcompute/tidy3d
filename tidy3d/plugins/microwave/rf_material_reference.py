"""Holds the reference materials for Tidy3D material library."""

from ...material_library.material_reference import ReferenceData

rf_material_refs = dict(
    Rogers3003=ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO3003™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3003-laminates",
    ),
    Rogers3010=ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO3010™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3010-laminates",
    ),
    Rogers4003C=ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO4003C™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates",
    ),
    Rogers4350B=ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO4350B™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates",
    ),
    ArlonAD255C=ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="AD255C High Performance Polyimide Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ad-series-laminates/ad255c-laminates",
    ),
    FR4_standard=ReferenceData(
        manufacturer="Isola",
        datasheet_title="Standard FR-4 Epoxy Glass Cloth Laminate",
        url="https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/",
    ),
    FR4_lowloss=ReferenceData(
        manufacturer="Isola",
        datasheet_title="Low loss FR-4 Epoxy Glass Cloth Laminate",
        url="https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/",
    ),
)
