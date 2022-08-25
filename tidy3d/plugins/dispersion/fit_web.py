"""Fit PoleResidue Dispersion models to optical NK data based on web service
"""
from typing import Tuple
from enum import Enum
import requests
from pydantic import PositiveInt, NonNegativeFloat, PositiveFloat, Field

from ...components.base import Tidy3dBaseModel
from ...components.types import Literal
from ...components import PoleResidue
from ...constants import MICROMETER, HERTZ
from ...log import log, WebError, Tidy3dError
from ...web.httputils import get_headers

from .fit import DispersionFitter

BOUND_MAX_FACTOR = 10


class AdvancedFitterParam(Tidy3dBaseModel):
    """Advanced fitter parameters"""

    bound_amp: NonNegativeFloat = Field(
        None,
        title="Upper bound of oscillator strength",
        description="Upper bound of real and imagniary part of oscillator "
        "strength ``c`` in the model :class:`.PoleResidue` (The default 'None' will trigger "
        "automatic setup based on the frequency range of interest).",
        units=HERTZ,
    )
    bound_f: NonNegativeFloat = Field(
        None,
        title="Upper bound of pole frequency",
        description="Upper bound of real and imaginary part of pole "
        "frequency ``a`` in the model :class:`.PoleResidue` (The default 'None' will trigger "
        "automatic setup based on the frequency range of interest).",
        units=HERTZ,
    )
    bound_eps_inf: float = Field(
        10.0,
        title="Upper bound of epsilon at infinity frequency",
        description="Upper bound of epsilon at infinity frequency. It must be no less than 1.",
        ge=1,
    )
    constraint: Literal["hard", "soft"] = Field(
        "hard",
        title="Type of constraint for stability",
        description="Stability constraint: 'hard' constraints are generally recommended since "
        "they are faster to compute per iteration, and they often require fewer iterations to "
        "converge since the search space is smaller. But sometimes the search space is "
        "so restrictive that all good solutions are missed, then please try the 'soft' constraints "
        "for larger search space. However, both constraints improve stability equally well.",
    )
    nlopt_maxeval: PositiveInt = Field(
        5000,
        title="Number of inner iterations",
        description="Number of iterations in each inner optimization.",
    )


# FitterData will be used internally
class FitterData(AdvancedFitterParam):
    """Data class for request body of Fitter where dipsersion data is input through tuple."""

    wvl_um: Tuple[float, ...] = Field(
        ...,
        title="Wavelengths",
        description="A set of wavelengths for dispersion data.",
        units=MICROMETER,
    )
    n_data: Tuple[float, ...] = Field(
        ...,
        title="Index of refraction",
        description="Real part of the complex index of refraction at each wavelength.",
    )
    k_data: Tuple[float, ...] = Field(
        None,
        title="Extinction coefficient",
        description="Imaginary part of the complex index of refraction at each wavelength.",
    )
    num_poles: PositiveInt = Field(
        1, title="Number of poles", description="Number of poles in model."
    )
    num_tries: PositiveInt = Field(
        50,
        title="Number of tries",
        description="Number of optimizations to run with different initial guess.",
    )
    tolerance_rms: NonNegativeFloat = Field(
        0.0,
        title="RMS error tolerance",
        description="RMS error below which the fit is successful and result is returned.",
    )
    bound_amp: PositiveFloat = Field(
        100.0,
        title="Upper bound of oscillator strength",
        description="Upper bound of oscillator strength in the model.",
        units="eV",
    )
    bound_f: PositiveFloat = Field(
        100.0,
        title="Upper bound of pole frequency",
        description="Upper bound of pole frequency in the model.",
        units="eV",
    )


URL_ENV = {
    "local": "http://127.0.0.1:8000",
    "dev": "https://tidy3d-service.dev-simulation.cloud",
    "prod": "https://tidy3d-service.simulation.cloud",
}


class ExceptionCodes(Enum):
    """HTTP exception codes to handle individually."""

    GATEWAY_TIMEOUT = 504
    NOT_FOUND = 404


class StableDispersionFitter(DispersionFitter):

    """Stable fitter based on web service"""

    @staticmethod
    def _set_url(config_env: Literal["default", "dev", "prod", "local"] = "default"):
        """Set the url of python web service

        Parameters
        ----------
        config_env : Literal["default", "dev", "prod", "local"], optional
            Service environment to pick from
        """

        _env = config_env
        if _env == "default":
            from ...web.config import DEFAULT_CONFIG  # pylint:disable=import-outside-toplevel

            _env = "dev" if "dev" in DEFAULT_CONFIG.web_api_endpoint else "prod"
        return URL_ENV[_env]

    @staticmethod
    def _setup_server(url_server: str):
        """set up web server access

        Parameters
        ----------
        url_server : str
            URL for the server
        """

        access_token = get_headers()
        headers = {"Authorization": access_token["Authorization"]}

        # test connection
        resp = requests.get(f"{url_server}/health")
        try:
            resp.raise_for_status()
        except Exception as e:
            raise WebError("Connection to the server failed. Please try again.") from e

        # test authorization
        resp = requests.get(f"{url_server}/health/access", headers=headers)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise WebError("Authorization to the server failed. Please try again.") from e

        return headers

    def _setup_webdata(
        self,
        num_poles: PositiveInt,
        num_tries: PositiveInt,
        tolerance_rms: NonNegativeFloat,
        advanced_param: AdvancedFitterParam,
    ) -> FitterData:
        """Setup FitterData to be provided to webservice

        Parameters
        ----------
        num_poles : PositiveInt
            Number of poles in the model.
        num_tries : PositiveInt
            Number of optimizations to run with random initial guess.
        tolerance_rms : NonNegativeFloat
            RMS error below which the fit is successful and the result is returned.
        advanced_param : :class:`AdvancedFitterParam`
            Other advanced parameters.

        Returns
        -------
        :class:`FitterData`
            Data class for request body of Fitter where dipsersion
            data is input through tuple.
        """

        # set up bound_f, bound_amp
        if advanced_param.bound_f is None:
            new_bound_f = self.frequency_range[1] * BOUND_MAX_FACTOR
            advanced_param = advanced_param.copy(update={"bound_f": new_bound_f})
        if advanced_param.bound_amp is None:
            new_bound_amp = self.frequency_range[1] * BOUND_MAX_FACTOR
            advanced_param = advanced_param.copy(update={"bound_amp": new_bound_amp})

        wvl_um, n_data, k_data = self._filter_wvl_range(
            wvl_min=self.wvl_range[0], wvl_max=self.wvl_range[1]
        )

        if self.lossy:
            k_data = k_data.tolist()
        else:
            k_data = None

        web_data = FitterData(
            wvl_um=wvl_um.tolist(),
            n_data=n_data.tolist(),
            k_data=k_data,
            num_poles=num_poles,
            num_tries=num_tries,
            tolerance_rms=tolerance_rms,
            bound_amp=self._Hz_to_eV(advanced_param.bound_amp),
            bound_f=self._Hz_to_eV(advanced_param.bound_f),
            bound_eps_inf=advanced_param.bound_eps_inf,
            constraint=advanced_param.constraint,
            nlopt_maxeval=advanced_param.nlopt_maxeval,
        )

        return web_data

    def fit(  # pylint:disable=arguments-differ, too-many-locals
        self,
        num_poles: PositiveInt = 1,
        num_tries: PositiveInt = 50,
        tolerance_rms: NonNegativeFloat = 1e-2,
        advanced_param: AdvancedFitterParam = AdvancedFitterParam(),
    ) -> Tuple[PoleResidue, float]:
        """Fits data a number of times and returns best results.

        Parameters
        ----------
        num_poles : PositiveInt, optional
            Number of poles in the model.
        num_tries : PositiveInt, optional
            Number of optimizations to run with random initial guess.
        tolerance_rms : NonNegativeFloat, optional
            RMS error below which the fit is successful and the result is returned.
        advanced_param : :class:`AdvancedFitterParam`, optional
            Other advanced parameters.

        Returns
        -------
        Tuple[:class:`.PoleResidue`, float]
            Best results of multiple fits: (dispersive medium, RMS error).
        """

        # get url
        url_server = self._set_url("default")
        headers = self._setup_server(url_server)

        # setup web_data
        web_data = self._setup_webdata(num_poles, num_tries, tolerance_rms, advanced_param)

        resp = requests.post(f"{url_server}/dispersion/fit", headers=headers, data=web_data.json())

        try:
            resp.raise_for_status()
        except Exception as e:
            if resp.status_code == ExceptionCodes.GATEWAY_TIMEOUT.value:
                raise Tidy3dError(
                    "Fitter failed due to timeout. Try to decrease "
                    "the number of tries, the number of inner iterations, "
                    "to relax RMS tolerance, or to use the 'hard' constraint."
                ) from e

            raise WebError(
                "Fitter failed. Try again, or tune the parameters, or contact us for more help."
            ) from e

        run_result = resp.json()
        best_medium = PoleResidue.parse_raw(run_result["message"])
        best_rms = float(run_result["rms"])

        if best_rms < tolerance_rms:
            log.info(f"\tfound optimal fit with RMS error = {best_rms:.2e}, returning")
        else:
            log.warning(
                f"\twarning: did not find fit "
                f"with RMS error under tolerance_rms of {tolerance_rms:.2e}"
            )
            log.info(f"\treturning best fit with RMS error {best_rms:.2e}")

        return best_medium, best_rms
