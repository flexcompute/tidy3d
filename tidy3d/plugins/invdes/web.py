# defines web API for running an inverse design optimization

import time

import pydantic.v1 as pd
import requests

from ...tidy3d.components.base import cached_property
from .base import InvdesBaseModel
from .optimizer import AdamOptimizer
from .result import InverseDesignResult

http = None


class Job(InvdesBaseModel):
    """Container that runs an `InverseDesign` optimization through the web API."""

    optimizer: AdamOptimizer = pd.Field(
        ..., title="Optimizer", description="Object defining the optimization to perform."
    )

    result_path: str = pd.Field(
        "results.hdf5",
        title="Result Path",
        description="File to store the inverse design result.",
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
        self.upload()
        self.start()
        self.monitor()
        return self.load()

    @cached_property
    def optimizer_id(self) -> str:
        """The task ID for this ``Job``. Uploads the ``Job`` if it hasn't already been uploaded."""
        return self._upload()

    def _upload(self) -> str:
        """Upload this job and return the task ID for handling."""
        body = self.json()
        resp = requests.post(body)
        optimizer_id = resp["optimizer_id"]
        return optimizer_id

    def upload(self) -> None:
        """Upload this ``Job``."""
        _ = self.optimizer_id

    def get_info(self) -> dict:
        """Return information about a :class:`Job`."""
        return requests.get(self.optimizer_id)

    @property
    def status(self):
        """Return current status of :class:`Job`."""
        return self.get_info().status

    def start(self) -> None:
        """Start running a :class:`Job`.

        Note
        ----
        To monitor progress of the :class:`Job`, call :meth:`Job.monitor` after started.
        """
        http.post()

    def monitor(self) -> None:
        """Monitor progress of running :class:`Job`.

        Note
        ----
        To load the output of completed simulation into :class:`.SimulationData` objects,
        call :meth:`Job.load`.
        """
        info_dict = {}
        while self.status != "finished":
            info_dict_new = self.get_info()
            if info_dict_new != info_dict:
                print(info_dict)
                info_dict = info_dict_new
            time.sleep(30)

    def download(self) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Note
        ----
        To load the data after download, use :meth:`Job.load`.
        """
        http.post(self.result_path)

    def load(self) -> InverseDesignResult:
        """Download job results and load them into a data object.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Returns
        -------
        Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            Object containing simulation results.
        """

    def delete(self) -> None:
        """Delete server-side data associated with :class:`Job`."""
        http.post(self.optimizer_id)

    def real_cost(self, verbose: bool = True) -> float:
        """Get the billed cost for the task associated with this job.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        float
            Billed cost of the task in FlexCredits.
        """
        return http.post()

    def estimate_cost(self, verbose: bool = True) -> float:
        """Compute the maximum FlexCredit charge for a given :class:`.Job`.

        Parameters
        ----------
        verbose : bool = True
            Whether to log the cost and helpful messages.

        Returns
        -------
        float
            Estimated cost of the task in FlexCredits.

        Note
        ----
        Cost is calculated assuming the simulation runs for
        the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.
        """
        return http.post()
