"""Project adapter for the upstream TurPy simulator."""

from TurPy import TurPy
from params import make_params


def make_turpy_simulator(
    grid_size=64,
    dx=0.03125,
    subharmonics=True,
    subharmonic_levels=3,
    wavelength=655e-9,
    n0=1.00027,
    outer_scale=30.0,
    inner_scale=5e-3,
    device=None,
):
    """Build a configured 2-D TurPy simulator for this project."""

    params = make_params(
        wavelength=wavelength,
        n=n0,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )
    params["field_size"] = (grid_size, grid_size)
    params["dx"] = dx
    params["subharmonics"] = subharmonics
    params["p"] = subharmonic_levels

    if device is not None:
        params["device"] = device

    simulator = TurPy(params)
    return params, simulator
