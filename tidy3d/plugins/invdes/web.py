# defines web API for running an inverse design optimization

import time

import pydantic.v1 as pd
import requests

from ...tidy3d.components.base import cached_property
from .base import InvdesBaseModel
from .optimizer import AdamOptimizer
from .result import InverseDesignResult

# time between status updates (seconds)
OPTIMIZATION_MONITOR_REFRESH = 15

url_base = "tidy3d/invdes/"

"""API

POST /tidy3d/invdes/run

    body:

        {
            optimizer : AdamOptimizer,
        }

    response:

        {
            optimizer_id : str,
        }

GET /tidy3d/invdes/{optimizer_id}/info

    body:
        {}

    response:
        {
            iteration : int,
            objective : float,
            gradient_norm : float,
            result : InverseDesignResult,
            status : str,
        }

PUT /tidy3d/invdes/{optimizer_id}/abort

    body:
        {}

    response:
        {}

GET /tidy3d/invdes/{optimizer_id}/load

    body:
        {}

    response:
        {
            result : inverseDesignResult,
        }


"""


class Job(InvdesBaseModel):
    """Container that runs an `InverseDesign` optimization through the web API."""

    optimizer: AdamOptimizer = pd.Field(
        ..., title="Optimizer", description="Object defining the optimization to perform."
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print status updates for the optimizer"
    )

    @pd.validator("optimizer", always=True)
    def _serializable_optimizer(cls, val):
        """Make sure the optimizer can be uploaded (defined with a serializable postprocess fn)."""

        try:
            val.json()
        except Exception as e:
            raise ValueError(
                "Could not serialize the 'optimizer'. For server-side optimization, "
                "the post-processing function must be defined "
                "inside of the 'InverseDesign.postprocess' field using built in expressions. "
            ) from e

        return val

    def run(self) -> InverseDesignResult:
        """Run optimization all the way through and return result"""
        self.submit()
        self.monitor()
        return self.load()

    @cached_property
    def optimizer_id(self) -> str:
        """The task ID for this ``Job``. Uploads the ``Job`` if it hasn't already been uploaded."""
        return self._submit()

    def _submit(self) -> str:
        """Upload this job and return the task ID for handling."""
        url = url_base + "submit"
        body = self.json
        resp = requests.post(url, data=body)
        return resp["optimizer_id"]

    def submit(self) -> None:
        """Upload this ``Job``."""
        _ = self.optimizer_id

    def get_info(self) -> dict:
        """Return information about a :class:`Job`."""
        url = url_base + f"{self.optimizer_id}/info"
        response = requests.get(url)
        return InverseDesignResult.parse_obj(response)

    @property
    def status(self):
        """Return current status of :class:`Job`."""
        info_dict = self.get_info()
        return info_dict["status"]

    @property
    def is_done(self) -> bool:
        """Is the run finished?"""
        return self.status in ("complete",)

    def monitor(self) -> None:
        """Monitor progress of running optimization."""

        info_dict = {}
        while not self.is_done:
            info_dict_new = self.get_info()
            if info_dict_new != info_dict:
                print(info_dict)
                info_dict = info_dict_new
            time.sleep(OPTIMIZATION_MONITOR_REFRESH)

    def load(self) -> InverseDesignResult:
        """Return results of optimization."""

        if not self.is_done:
            raise ValueError(f"Optimization is not finished, status of '{self.status}'.")

        url = url_base + f"{self.optimizer_id}/load"
        response = requests.get(url)
        return InverseDesignResult.parse_obj(response)

    def abort(self) -> None:
        """Interrupt server-side optimization associated with this :class:`Job`."""
        url = url_base + f"{self.optimizer_id}/abort"
        requests.put(url)
