"""Holds the reference materials for Tidy3D material library."""

from __future__ import annotations

from tidy3d.material_library.material_reference import ReferenceData

rf_material_refs = {
    "Rogers3003": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO3003™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3003-laminates",
    ),
    "Rogers3010": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO3010™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro3000-series-laminates/ro3010-laminates",
    ),
    "Rogers4003C": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO4003C™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates",
    ),
    "Rogers4350B": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RO4350B™ Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4350b-laminates",
    ),
    "ArlonAD255C": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="AD255C High Performance Polyimide Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/ad-series-laminates/ad255c-laminates",
    ),
    "FR4_standard": ReferenceData(
        manufacturer="Isola",
        datasheet_title="Standard FR-4 Epoxy Glass Cloth Laminate",
        url="https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/",
    ),
    "FR4_lowloss": ReferenceData(
        manufacturer="Isola",
        datasheet_title="Low loss FR-4 Epoxy Glass Cloth Laminate",
        url="https://www.isola-group.com/pcb-laminates-prepreg/is410-fr-4-epoxy-laminate-and-prepreg/",
    ),
    "RogersRT_duroid_5880": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RT/duroid 5870/5880 High frequency Laminates",
        url="https://www.rogerscorp.com/advanced-electronics-solutions/rt-duroid-laminates/rt-duroid-5880-laminates",
    ),
    "RogersRT_duroid_6035HTC": ReferenceData(
        manufacturer="Rogers Corporation",
        datasheet_title="RT/duroid® 6035HTC High Frequency Laminate",
        url="https://www.rogerscorp.com/-/media/project/rogerscorp/documents/advanced-electronics-solutions/english/data-sheets/rt-duroid-6035htc-high-frequency-laminates.pdf",
    ),
    "PTFE": ReferenceData(
        manufacturer="Various",
        datasheet_title="HIGH PERFORMANCE MICROWAVE INTERCONNECT PRODUCTS",
        url="https://configurator.teledynestorm.com/pdf/DielectricOptions.pdf",
    ),
    "Alumina_kyocera": ReferenceData(
        manufacturer="Kyocera Global",
        datasheet_title="AO700/AO800 (High Flexural Strength Aluminum Oxide)",
        url="https://global.kyocera.com/prdct/semicon/search_material/detail/ao700_ao800.html",
    ),
    "Alumina_AO479U": ReferenceData(
        manufacturer="Kyocera Global",
        datasheet_title="Low Dielectric Loss Tangent, High Strength, and High Purity Alumina (AO479U)",
        url="https://global.kyocera.com/prdct/fc/technologies/013.html",
    ),
    "Getek": ReferenceData(
        manufacturer="Isola Group",
        datasheet_title="Getek Low Dk/Df Laminate and Prepreg",
        url="https://www.midwestpcb.com/data_sheets/GETEK.pdf",
    ),
    "Isola_370HR": ReferenceData(
        manufacturer="Isola Group",
        datasheet_title="Isola 370HR Laminate and Prepreg",
        url="https://www.isola-group.com/wp-content/uploads/data-sheets/370hr.pdf",
    ),
    "Isola_FR406": ReferenceData(
        manufacturer="Isola Group",
        datasheet_title="FR406 Standard Loss, High Tg Epoxy Laminate and Prepreg",
        url="https://www.isola-group.com/wp-content/uploads/data-sheets/fr406-laminate-and-prepreg.pdf",
    ),
    "Isola_FR408": ReferenceData(
        manufacturer="Isola Group",
        datasheet_title="FR408 Standard Loss, High Tg Epoxy Laminate and Prepreg",
        url="https://www.isola-group.com/wp-content/uploads/data-sheets/fr408.pdf",
    ),
    "Megtron6_R5775_R5670(KG)": ReferenceData(
        manufacturer="Panasonic Industry",
        datasheet_title="Ultra-low transmission loss Highly heat resistant "
        "Multi-layer circuit board materials",
        url="https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/261273",
    ),
    "Megtron6_R5775_R5670(N)": ReferenceData(
        manufacturer="Panasonic Industry",
        datasheet_title="High Speed, Low Loss Multi-layer Materials",
        url="https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/244766",
    ),
    "Nelco_N4000-6": ReferenceData(
        manufacturer="Park Electrochemical Corp.",
        datasheet_title="Park Electrochemical Nelco® N4000-6 FC Multifunctional Epoxy Laminate and Prepreg",
        url="https://www.lookpolymers.com/pdf/Park-Electrochemical-Nelco-N4000-6-FC-Multifunctional-Epoxy-Laminate-and-Prepreg.pdf",
    ),
    "Nelco_N4000-13": ReferenceData(
        manufacturer="Park Electrochemical Corp.",
        datasheet_title="High-Speed Multifunctional Epoxy Laminate & Prepreg",
        url="https://www.cirexx.com/wp-content/uploads/n4000-13ep-1.pdf",
    ),
    "Matula": ReferenceData(
        journal="Matula, R.A. Journal of Physical and Chemical Reference Data. 8 (4): 1147 (1979).",
        url="https://srd.nist.gov/JPCRD/jpcrd155.pdf",
        doi="https://doi.org/10.1063/1.555614",
    ),
    "Annealed_Copper": ReferenceData(
        journal="Copper wire tables, U.S. National Bureau of Standards, (1914).",
        url="https://archive.org/details/copperwiretables31unituoft/page/10/mode/2up",
    ),
    "Alum_293K": ReferenceData(
        journal="P.D. Desai, H.M. James, and C.Y. Ho. Electrical Resistivity of Aluminum and Manganese, "
        "Journal of Physical and Chemical Reference Data, Vol. 13, No. 4 (1984)",
        url="https://srd.nist.gov/JPCRD/jpcrd260.pdf",
    ),
    "Brass": ReferenceData(
        manufacturer="Copper Development Association",
        datasheet_title="Electronic Connector Design Guide: Conductivity of Brass",
        url="https://www.copper.org/applications/industrial/DesignGuide/selection/conductbrass02.html",
    ),
    "Zinc": ReferenceData(
        journal="P.D. Desai, Electrical Resistivity of Selected Elements., "
        "Journal of Physical and Chemical Reference Data, Vol. 13, No. 4 (1984).",
        url="https://pubs.aip.org/aip/jpr/article-abstract/13/4/1069/241360/Electrical-Resistivity-of-Selected-Elements",
    ),
    "CRC_Handbook": ReferenceData(
        journal="Rumble, J. R. (Ed.). (2024), CRC Handbook of Chemistry and Physics (105th ed.)., CRC Press",
        url="https://www.engineeringtoolbox.com/conductors-d_1381.html",
    ),
    "Raymond_Serway": ReferenceData(
        journal="Raymond A. Serway (1998)., Principles of Physics, "
        "Fort Worth, Texas; London: Saunders College Pub. p. 602",
        url="https://archive.org/details/principlesofphys00serw/page/602/mode/2up",
    ),
    "AISI_1008": ReferenceData(
        journal="Oberg, E., Jones, F. D., Horton, H. L., & Ryffel, H. H. (2020)., "
        "Machinery's Handbook (31st ed.). Industrial Press."
    ),
}
