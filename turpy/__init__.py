"""Project adapter for the upstream TurPy simulator."""

from TurPy import TurPy
from params import make_params


def make_turpy_simulator(
    grid_size=64,
    dx=20e-6,
    subharmonics=True,
    device=None,
):
    """Build a configured 2-D TurPy simulator for this project."""

    params = make_params()
    params["field_size"] = (grid_size, grid_size)
    params["dx"] = dx
    params["subharmonics"] = subharmonics

    if device is not None:
        params["device"] = device

    simulator = TurPy(params)
    return params, simulator
