from ib_tools.bracket_legs import FixedStop
from ib_tools.execution_models import OcaExecModel


def test_oca_group():
    e = OcaExecModel(stop=FixedStop(1))
    oca_group = e.oca_group()
    assert isinstance(oca_group, str)
    assert len(oca_group) > 10
    assert oca_group.endswith("00000")
