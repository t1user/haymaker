import haymaker.supervisor as supervisor_package
from haymaker.supervisor import supervisor


def test_package_exports_connection_supervisor() -> None:
    """The package exports the connection supervisor and shared settings."""

    assert (
        supervisor_package.ConnectionSupervisor is supervisor.ConnectionSupervisor
    )
    assert supervisor_package.ConnectionSettings is supervisor.ConnectionSettings
