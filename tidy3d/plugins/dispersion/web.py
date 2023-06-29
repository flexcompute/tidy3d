"""Fit PoleResidue Dispersion models to optical NK data based on web service"""

from __future__ import annotations

import ssl
from typing import Tuple, Optional
from enum import Enum
import requests
import pydantic
from pydantic import PositiveInt, NonNegativeFloat, PositiveFloat, Field, validator

from ...log import log
from ...components.base import Tidy3dBaseModel
from ...components.types import Literal
from ...components.medium import PoleResidue
from ...constants import MICROMETER, HERTZ
from ...exceptions import WebError, Tidy3dError, SetupError
from ...web.httputils import get_headers
from ...web.environment import Env

from .fit import DispersionFitter


BOUND_MAX_FACTOR = 10

URL_ENV = {
    "local": "http://127.0.0.1:8000",
    "dev": "https://tidy3d-service.dev-simulation.cloud",
    "prod": "https://tidy3d-service.simulation.cloud",
}


class ExceptionCodes(Enum):
    """HTTP exception codes to handle individually."""

    GATEWAY_TIMEOUT = 504
    NOT_FOUND = 404


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
        description="Upper bound of real and imaginary part of ``a`` that corresponds to pole "
        "damping rate and frequency in the model :class:`.PoleResidue` (The default 'None' "
        "will trigger automatic setup based on the frequency range of interest).",
        units=HERTZ,
    )
    bound_f_lower: NonNegativeFloat = Field(
        0.0,
        title="Lower bound of pole frequency",
        description="Lower bound of imaginary part of ``a`` that corresponds to pole "
        "frequency in the model :class:`.PoleResidue`.",
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
    random_seed: Optional[int] = Field(
        0,
        title="Random seed for starting coefficients",
        description="The fitting tool performs global optimizations with random "
        "starting coefficients. With the same random seed, one obtains identical "
        "results when re-running the fitter; on the other hand, if "
        "one wants to re-run the fitter several times to obtain the best results, "
        "the value of the seed should be changed, or set to  ``None`` so that "
        "the starting coefficients are different each time. ",
        ge=0,
        lt=2**32,
    )

    @validator("bound_f_lower", always=True)
    def _validate_lower_frequency_bound(cls, val, values):
        """bound_f_lower cannot be larger than bound_f."""
        if values["bound_f"] is not None and val > values["bound_f"]:
            raise SetupError(
                "The upper bound 'bound_f' cannot be smaller "
                "than the lower bound 'bound_f_lower'."
            )
        return val


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

    @staticmethod
    def create(
        fitter: DispersionFitter,
        num_poles: PositiveInt,
        num_tries: PositiveInt,
        tolerance_rms: NonNegativeFloat,
        advanced_param: AdvancedFitterParam,
    ) -> FitterData:
        """Setup FitterData to be provided to web service

        Parameters
        ----------
        fitter : DispersionFitter
            Fitter with the data to fit.
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
            Data class for request body of Fitter where dispersion
            data is input through tuple.
        """

        # set up bound_f, bound_amp
        if advanced_param.bound_f is None:
            new_bound_f = (
                advanced_param.bound_f_lower + fitter.frequency_range[1] * BOUND_MAX_FACTOR
            )
            advanced_param = advanced_param.copy(update={"bound_f": new_bound_f})
        if advanced_param.bound_amp is None:
            new_bound_amp = fitter.frequency_range[1] * BOUND_MAX_FACTOR
            advanced_param = advanced_param.copy(update={"bound_amp": new_bound_amp})

        wvl_um, n_data, k_data = fitter.data_in_range

        if fitter.lossy:
            k_data = k_data.tolist()
        else:
            k_data = None

        # pylint: disable=protected-access
        task = FitterData(
            wvl_um=wvl_um.tolist(),
            n_data=n_data.tolist(),
            k_data=k_data,
            num_poles=num_poles,
            num_tries=num_tries,
            tolerance_rms=tolerance_rms,
            bound_amp=fitter._Hz_to_eV(advanced_param.bound_amp),
            bound_f=fitter._Hz_to_eV(advanced_param.bound_f),
            bound_f_lower=fitter._Hz_to_eV(advanced_param.bound_f_lower),
            bound_eps_inf=advanced_param.bound_eps_inf,
            constraint=advanced_param.constraint,
            nlopt_maxeval=advanced_param.nlopt_maxeval,
            random_seed=advanced_param.random_seed,
        )
        return task

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
            _env = "dev" if "dev" in Env.current.web_api_endpoint else "prod"
        return URL_ENV[_env]

    @staticmethod
    def _setup_server(url_server: str):
        """set up web server access

        Parameters
        ----------
        url_server : str
            URL for the server
        """

        try:
            # test connection
            resp = requests.get(f"{url_server}/health", verify=Env.current.ssl_verify)
            resp.raise_for_status()
        except (requests.exceptions.SSLError, ssl.SSLError):
            log.info("Retrying with SSL verification disabled.")
            Env.current.ssl_verify = False
            resp = requests.get(f"{url_server}/health", verify=Env.current.ssl_verify)
        except Exception as e:
            raise WebError("Connection to the server failed. Please try again.") from e

        return get_headers()

    def run(self) -> Tuple[PoleResidue, float]:
        """Execute the data fit using the stable fitter in the server.

        Returns
        -------
        Tuple[:class:`.PoleResidue`, float]
            Best results of multiple fits: (dispersive medium, RMS error).
        """

        url_server = self._set_url("default")
        headers = self._setup_server(url_server)

        resp = requests.post(
            f"{url_server}/dispersion/fit",
            headers=headers,
            data=self.json(),
            verify=Env.current.ssl_verify,
        )

        try:
            resp.raise_for_status()
        except Exception as e:
            if resp.status_code == ExceptionCodes.GATEWAY_TIMEOUT.value:
                msg = (
                    (
                        "Fitter failed due to timeout. Try to decrease the number of tries or "
                        "inner iterations, to relax the RMS tolerance, or to use the 'hard' "
                        "constraint."
                    )
                    if self.constraint != "hard"
                    else (
                        "Fitter failed due to timeout. Try to decrease the number of tries or "
                        "inner iterations, or to relax the RMS tolerance."
                    )
                )
                raise Tidy3dError(msg) from e

            raise WebError(
                "Fitter failed. Try again, tune the parameters, or contact us for more help."
            ) from e

        run_result = resp.json()
        best_medium = PoleResidue.parse_raw(run_result["message"])
        best_rms = float(run_result["rms"])

        if best_rms < self.tolerance_rms:
            log.info("Found optimal fit with RMS error %.3g", best_rms)
        else:
            log.warning(
                "Unable to fit with RMS error under 'tolerance_rms' of %.3g", self.tolerance_rms
            )
            log.info("Returning best fit with RMS error %.3g", best_rms)

        return best_medium, best_rms


def run(
    fitter: DispersionFitter,
    num_poles: PositiveInt = 1,
    num_tries: PositiveInt = 50,
    tolerance_rms: NonNegativeFloat = 1e-2,
    advanced_param: AdvancedFitterParam = AdvancedFitterParam(),
) -> Tuple[PoleResidue, float]:
    """Execute the data fit using the stable fitter in the server.

    Parameters
    ----------
    fitter : DispersionFitter
        Fitter with the data to fit.
    num_poles : PositiveInt, optional
        Number of poles in the model.
    num_tries : PositiveInt, optional
        Number of optimizations to run with random initial guess.
    tolerance_rms : NonNegativeFloat, optional
        RMS error below which the fit is successful and the result is returned.
    advanced_param : :class:`AdvancedFitterParam`, optional
        Advanced parameters passed on to the server.

    Returns
    -------
    Tuple[:class:`.PoleResidue`, float]
        Best results of multiple fits: (dispersive medium, RMS error).
    """
    task = FitterData.create(fitter, num_poles, num_tries, tolerance_rms, advanced_param)
    return task.run()


class StableDispersionFitter(DispersionFitter):
    """Deprecated."""

    @pydantic.root_validator()
    def _deprecate_stable_fitter(cls, values):
        log.warning(
            "'StableDispersionFitter' has been deprecated. Use 'DispersionFitter' with "
            "'tidy3d.plugins.dispersion.web.run' to access the stable fitter from the web server."
        )
        return values

    def fit(  # pylint:disable=arguments-differ, too-many-locals
        self,
        num_poles: PositiveInt = 1,
        num_tries: PositiveInt = 50,
        tolerance_rms: NonNegativeFloat = 1e-2,
        advanced_param: AdvancedFitterParam = AdvancedFitterParam(),
    ) -> Tuple[PoleResidue, float]:
        """Deprecated."""
        return run(self, num_poles, num_tries, tolerance_rms, advanced_param)
