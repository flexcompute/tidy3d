"""Holds the reference materials for Tidy3D material library."""
import pydantic as pd

from ..components.base import Tidy3dBaseModel


class ReferenceData(Tidy3dBaseModel):
    """Reference data."""

    doi: str = pd.Field(None, title="DOI", description="DOI of the reference.")
    journal: str = pd.Field(
        None,
        title="Journal publication info",
        description="Publication info in the order of author, title, journal volume, and year.",
    )
    url: str = pd.Field(
        None,
        title="URL link",
        description="Some reference can be accessed through a url link to its pdf etc.",
    )


material_refs = dict(
    Li2014=ReferenceData(
        journal="Y. Li, A. Chernikov, X. Zhang, A. Rigosi, H. M. Hill, A. M. van der Zande, "
        "D. A. Chenet, E. Shih, J. Hone, and T. F. Heinz. Measurement of the optical dielectric "
        "function of monolayer transition-metal dichalcogenides: MoS2, MoSe2, WS2, and WSe2, "
        "Phys. Rev. B 90, 205422 (2014)",
        doi="https://doi.org/10.1103/PhysRevB.90.205422",
    ),
    Yang2015=ReferenceData(
        journal="H. U. Yang, J. D'Archangel, M. L. Sundheimer, E. Tucker, G. D. Boreman, "
        "M. B. Raschke. Optical dielectric function of silver, Phys. Rev. B 91, 235137 (2015)",
        doi="https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.235137",
    ),
    Olmon2012=ReferenceData(
        journal="R. L. Olmon, B. Slovick, T. W. Johnson, D. Shelton, S.-H. Oh, "
        "G. D. Boreman, and M. B. Raschke. Optical dielectric function of "
        "gold, Phys. Rev. B 86, 235147 (2012)",
        doi="https://doi.org/10.1103/PhysRevB.86.235147",
    ),
    Rakic1995=ReferenceData(
        journal="A. D. Rakic. Algorithm for the determination of intrinsic optical "
        "constants of metal films: application to aluminum, Appl. Opt. 34, 4755-4767 (1995)",
        doi="https://doi.org/10.1364/AO.34.004755",
    ),
    Rakic1998=ReferenceData(
        journal="A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski. "
        "Optical properties of metallic films for vertical-cavity optoelectronic "
        "devices, Appl. Opt. 37, 5271-5283 (1998)",
        doi="https://doi.org/10.1364/AO.37.005271",
    ),
    JohnsonChristy1972=ReferenceData(
        journal="P. B. Johnson and R. W. Christy. Optical constants of the noble "
        "metals, Phys. Rev. B 6, 4370-4379 (1972)",
        doi="https://doi.org/10.1103/PhysRevB.6.4370",
    ),
    Horiba=ReferenceData(
        journal="Horiba Technical Note 08: Lorentz Dispersion Model",
        url="http://www.horiba.com/fileadmin/uploads/Scientific/Downloads"
        "/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf",
    ),
    FernOnton1971=ReferenceData(
        journal="R. E. Fern and A. Onton. Refractive index of AlAs, "
        "J. Appl. Phys. 42, 3499-3500 (1971)",
        doi="https://doi.org/10.1063/1.1660760",
    ),
    Sultanova2009=ReferenceData(
        journal="N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties "
        "of optical polymers, Acta Physica Polonica A 116, 585-587 (2009)",
        doi="https://doi.org/10.12693/aphyspola.116.585",
    ),
    Malitson1965=ReferenceData(
        journal="I. H. Malitson. Interspecimen comparison of the refractive "
        "index of fused silica, J. Opt. Soc. Am. 55, 1205-1208 (1965)",
        doi="https://doi.org/10.1364/JOSA.55.001205",
    ),
    Tan1998=ReferenceData(
        journal="C. Z. Tan. Determination of refractive index of silica glass "
        "for infrared wavelengths by IR spectroscopy, J. Non-Cryst. Solids 223, 158-163 (1998)",
        doi="https://doi.org/10.1016/S0022-3093(97)00438-9",
    ),
    Skauli2003=ReferenceData(
        journal="T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, "
        "O. Levi, L. A. Eyres, J. S. Harris, M. M. Fejer, B. Gerard, "
        "L. Becouarn, and E. Lallier. Improved dispersion relations "
        "for GaAs and applications to nonlinear optics, J. Appl. Phys., 94, 6447-6455 (2003)",
        doi="https://doi.org/10.1063/1.1621740",
    ),
    Icenogle1976=ReferenceData(
        journal="H. W. Icenogle, Ben C. Platt, and William L. Wolfe. "
        "Refractive indexes and temperature coefficients of germanium "
        "and silicon Appl. Opt. 15 2348-2351 (1976)",
        doi="https://doi.org/10.1364/AO.15.002348",
    ),
    Barnes1979=ReferenceData(
        journal="N. P. Barnes and M. S. Piltch. Temperature-dependent "
        "Sellmeier coefficients and nonlinear optics average power limit "
        "for germanium J. Opt. Soc. Am. 69 178-180 (1979)",
        doi="https://doi.org/10.1364/JOSA.69.000178",
    ),
    Pettit1965=ReferenceData(
        journal="G. D. Pettit and W. J. Turner. Refractive index of InP, "
        "J. Appl. Phys. 36, 2081 (1965)",
        doi="https://doi.org/10.1063/1.1714410",
    ),
    Pikhtin1978=ReferenceData(
        journal="A. N. Pikhtin and A. D. Yas'kov. Disperson of the "
        "refractive index of semiconductors with diamond and zinc-blende "
        "structures, Sov. Phys. Semicond. 12, 622-626 (1978)",
    ),
    HandbookOptics=ReferenceData(
        journal="Handbook of Optics, 2nd edition, Vol. 2. McGraw-Hill 1994 (ISBN 9780070479746)",
    ),
    StephensMalitson1952=ReferenceData(
        journal="R. E. Stephens and I. H. Malitson. Index of refraction of "
        "magnesium oxide, J. Res. Natl. Bur. Stand. 49 249-252 (1952)",
        doi="https://doi.org/10.6028/jres.049.025",
    ),
    Werner2009=ReferenceData(
        journal="W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. "
        "Optical constants and inelastic electron-scattering data for 17 "
        "elemental metals, J. Phys Chem Ref. Data 38, 1013-1092 (2009)",
        doi="https://doi.org/10.1063/1.3243762",
    ),
    Luke2015=ReferenceData(
        journal="K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson. "
        "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator, "
        "Opt. Lett. 40, 4823-4826 (2015)",
        doi="https://doi.org/10.1364/OL.40.004823",
    ),
    Philipp1973=ReferenceData(
        journal="H. R. Philipp. Optical properties of silicon nitride, "
        "J. Electrochim. Soc. 120, 295-300 (1973)",
        doi="https://doi.org/10.1149/1.2403440",
    ),
    Baak1982=ReferenceData(
        journal="T. Baak. Silicon oxynitride; a material for GRIN optics, "
        "Appl. Optics 21, 1069-1072 (1982)",
        doi="https://doi.org/10.1364/AO.21.001069",
    ),
    Nigara1968=ReferenceData(
        journal="Y. Nigara. Measurement of the optical constants of yttrium oxide, "
        "Jpn. J. Appl. Phys. 7, 404-408 (1968)",
        doi="https://doi.org/10.1143/JJAP.7.404",
    ),
    Zelmon1998=ReferenceData(
        journal="D. E. Zelmon, D. L. Small and R. Page. Refractive-index measurements "
        "of undoped yttrium aluminum garnet from 0.4 to 5.0 μm, Appl. Opt. 37, 4933-4935 (1998)",
        doi="https://doi.org/10.1364/AO.37.004933",
    ),
    SalzbergVilla1957=ReferenceData(
        journal="C. D. Salzberg and J. J. Villa. Infrared Refractive Indexes of "
        "Silicon, Germanium and Modified Selenium Glass, J. Opt. Soc. Am., 47, 244-246 (1957)",
        doi="https://doi.org/10.1364/JOSA.47.000244",
    ),
    Tatian1984=ReferenceData(
        journal="B. Tatian. Fitting refractive-index data with the Sellmeier "
        "dispersion formula, Appl. Opt. 23, 4477-4485 (1984)",
        doi="https://doi.org/10.1364/AO.23.004477",
    ),
    Li1993_293K=ReferenceData(
        journal="H. H. Li. Refractive index of silicon and germanium and its wavelength "
        "and temperature derivatives, J. Phys. Chem. Ref. Data 9, 561-658 (1993)",
        doi="https://doi.org/10.1063/1.555624",
    ),
    Green2008=ReferenceData(
        journal="M. A. Green. Self-consistent optical parameters of intrinsic silicon "
        "at 300K including temperature coefficients, Sol. Energ. Mat. "
        "Sol. Cells 92, 1305–1310 (2008)",
        doi="https://doi.org/10.1016/j.solmat.2008.06.009",
    ),
    Zemax=ReferenceData(
        journal="SCHOTT Zemax catalog 2017-01-20b",
        url="https://refractiveindex.info/download/data/2017/schott_2017-01-20.pdf",
    ),
    Hanson2008=ReferenceData(
        journal="George W. Hanson. Dyadic Green's Functions for an Anisotropic, "
        "Non-Local Model of Biased Graphene, IEEE Trans. Antennas Propag. 56, 3, 747-757 (2008)",
        doi="https://doi.org/10.1109/TAP.2008.917005",
    ),
)
