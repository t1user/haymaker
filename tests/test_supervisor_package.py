import pytest

import haymaker.supervisor as supervisor_package
from haymaker.supervisor import supervisor as state_supervisor
from haymaker.supervisor import supervisor_one as onion_supervisor


def test_package_exports_state_supervisor_by_default() -> None:
    """The public package import keeps the production default implementation."""

    assert (
        supervisor_package.ConnectionSupervisor is state_supervisor.ConnectionSupervisor
    )
    assert supervisor_package.ConnectionSettings is state_supervisor.ConnectionSettings


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "state"),
        ({"supervisor": "onion"}, "onion"),
        ({"app": {"supervisor": "onion"}}, "onion"),
        ({"supervisor": "onion", "app": {"supervisor": "state"}}, "state"),
    ],
)
def test_configured_supervisor_mode(config: dict, expected: str) -> None:
    """Supervisor selection supports live nested and dataloader flat config."""

    assert supervisor_package._configured_supervisor_mode(config) == expected


def test_configured_supervisor_mode_rejects_unknown_name() -> None:
    """Invalid supervisor names fail during package-level selection."""

    with pytest.raises(ValueError, match="expected 'state' or 'onion'"):
        supervisor_package._configured_supervisor_mode(
            {"app": {"supervisor": "state-machine"}}
        )


def test_concrete_supervisors_share_settings_type() -> None:
    """Both implementations use the same public settings dataclass."""

    assert state_supervisor.ConnectionSettings is supervisor_package.ConnectionSettings
    assert onion_supervisor.ConnectionSettings is supervisor_package.ConnectionSettings
