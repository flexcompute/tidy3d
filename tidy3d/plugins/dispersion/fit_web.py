"""Fit PoleResidue Dispersion models to optical NK data based on web service
"""
from typing import Tuple, List
import numpy as np
import requests
from pydantic import BaseModel, PositiveInt, NonNegativeFloat, PositiveFloat, Field

from ...components.types import Literal
from ...components import PoleResidue
from ...constants import MICROMETER
from ...log import log
from .fit import DispersionFitter


class FitterData(BaseModel):
    """Data class for request body of Fitter where dipsersion data is input through list"""

    wvl_um: List[float] = Field(
        ...,
        title="Wavelengths",
        description="A list of wavelengths for dispersion data.",
        units=MICROMETER,
    )
    n_data: List[float] = Field(
        ...,
        title="Index of refraction",
        description="Real part of the complex index of refraction.",
    )
    k_data: List[float] = Field(
        ...,
        title="Extinction coefficient",
        description="Imaginary part of the complex index of refraction.",
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
        unis="eV",
    )
    bound_f: PositiveFloat = Field(
        100.0,
        title="Upper bound of pole frequency",
        description="Upper bound of pole frequency in the model.",
        units="eV",
    )
    constraint: Literal["hard", "soft"] = Field(
        "hard",
        title="Type of constraint for stability",
        description="Stability constraint enfored on each pole (hard),"
        " or the summed contribution (soft).",
    )
    nlopt_maxeval: PositiveInt = Field(
        1000,
        title="Number of inner iterations",
        description="Number of iterations in each optimization",
    )


URL_ENV = {
    "local": "http://127.0.0.1:8000/dispersion/fit",
    "dev": "https://tidy3d-service.dev-simulation.cloud/dispersion/fit",
    "prod": "https://tidy3d-service.simulation.cloud/dispersion/fit",
}


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
        if config_env == "default":
            from ...web.config import DEFAULT_CONFIG  # pylint:disable=import-outside-toplevel

            if "dev" in DEFAULT_CONFIG.web_api_endpoint:
                _env = "dev"
            else:
                _env = "prod"

        return URL_ENV[_env]

    def fit(  # pylint:disable=arguments-differ, too-many-arguments, too-many-locals
        self,
        num_poles: PositiveInt = 1,
        num_tries: PositiveInt = 50,
        tolerance_rms: NonNegativeFloat = 0.0,
        bound_amp: PositiveFloat = DispersionFitter.eV_to_Hz(100.0),
        bound_f: PositiveFloat = DispersionFitter.eV_to_Hz(100.0),
        constraint: Literal["hard", "soft"] = "hard",
        nlopt_maxeval: PositiveInt = 1000,
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
        bound_amp : PositiveFloat, optional
            Bound on the amplitude of poles, namely on Re[c], Im[c], Re[a]
        bound_f : PositiveFloat, optional
            Bound on the frequency of poles, namely on Im[a]. Unit: Hz
        constraint : Literal["hard", "soft"], optional
            'hard' or 'soft'
        nlopt_maxeval : PositiveInt, optional
            Maxeval in each optimization

        Returned
        ------------------
        Tuple[``PoleResidue``, float]
            Best results of multiple fits: (dispersive medium, RMS error).
        """

        from ...web.auth import (
            get_credentials,
        )  # pylint:disable=import-outside-toplevel, unused-import
        from ...web.httputils import get_headers  # pylint:disable=import-outside-toplevel

        _url = self._set_url("default")

        # get_credentials()
        access_token = get_headers()
        headers = {"Authorization": access_token["Authorization"]}

        web_data = FitterData(
            wvl_um=self.wvl_um.tolist(),
            n_data=self.n_data.tolist(),
            k_data=self.k_data.tolist(),
            num_poles=num_poles,
            num_tries=num_tries,
            tolerance_rms=tolerance_rms,
            bound_amp=self.Hz_to_eV(bound_amp),
            bound_f=self.Hz_to_eV(bound_f),
            constraint=constraint,
            nlopt_maxeval=nlopt_maxeval,
        )

        r_post = requests.post(
            _url,
            headers=headers,
            data=web_data.json(),
        )

        try:
            run_result = r_post.json()
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

        except Exception as e:  # pylint:disable=broad-except
            log.warning(e)
            log.error(
                "Fitter failed due to timeout. Try to decrease "
                "the number of tries, the number of internal iterations, "
                "or relax RMS tolerance."
            )
            return PoleResidue(), np.inf
