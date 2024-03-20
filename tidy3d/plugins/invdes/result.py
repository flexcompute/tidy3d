# container for everything defining the inverse design

import typing


import tidy3d as td


class OptimizeResult(td.components.base.Tidy3dBaseModel):
    """Container for the result of an ``InverseDesign.run()`` call."""

    history: typing.Dict[str, typing.Any]

    @property
    def final(self) -> typing.Dict[str, typing.Any]:
        """Dictionary of final values in ``self.history``."""
        return {key: value[-1] for key, value in self.history.items()}

    def get_final(self, key: str) -> typing.Any:
        """Get the final value of a field in the ``self.history`` by key."""
        return self.final.get(key)
