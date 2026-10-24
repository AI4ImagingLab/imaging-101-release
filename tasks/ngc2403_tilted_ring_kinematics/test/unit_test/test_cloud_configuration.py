"""Selected configuration, not Galmod's lower-level constructor default."""
import inspect
from src.physics_model import cloud_cube


def test_author_selected_default_cloud_density():
    # Frozen source GALMOD_PAR::CDENS=10; actual Galmod::input forwards it.
    # The separate low-level Galmod constructor's 1 is not this configuration.
    assert inspect.signature(cloud_cube).parameters['cdens'].default==10
