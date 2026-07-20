from typing import cast
from unittest.mock import Mock

import ib_insync as ibi
from runtime_helpers import AtomRuntimeHarness

from haymaker.base import Atom
from haymaker.contract_registry import ContractRegistry
from haymaker.controller import Controller
from haymaker.datastore import FrameStoreProvider


def test_default_atom_runtime_installs_core_services(atom_runtime) -> None:
    atom = Atom()

    assert atom.ib is atom_runtime.ib
    assert atom.sm is atom_runtime.sm
    assert atom.contract_registry is atom_runtime.contract_registry
    assert atom_runtime.trader is not None
    assert atom_runtime.controller is None


def test_atom_runtime_factory_respects_custom_services(
    atom_runtime_factory, state_machine
) -> None:
    ib = ibi.IB()
    registry = ContractRegistry()
    controller = cast(Controller, object())
    frame_store_provider = Mock(spec=FrameStoreProvider)

    runtime = atom_runtime_factory(
        ib=ib,
        sm=state_machine,
        contract_registry=registry,
        controller=controller,
        frame_store_provider=frame_store_provider,
    )
    atom = Atom()

    assert runtime.ib is ib
    assert runtime.sm is state_machine
    assert runtime.contract_registry is registry
    assert runtime.controller is controller
    assert runtime.frame_store_provider is frame_store_provider
    assert atom.ib is ib
    assert atom.contract_registry is registry


def test_atom_runtime_records_restart_requests(atom_runtime) -> None:
    atom = Atom()

    assert atom.request_restart is not None
    assert atom.request_restart("manual restart")

    assert atom_runtime.restart_requests == ["manual restart"]


def test_atom_runtime_bind_controller_updates_installed_runtime(atom_runtime) -> None:
    controller = cast(Controller, object())

    result = atom_runtime.bind_controller(controller)

    assert result is controller
    assert Atom().runtime.controller is controller


def test_atom_runtime_factory_replaces_previous_runtime(
    atom_runtime_factory, state_machine
) -> None:
    first = atom_runtime_factory(ib=ibi.IB(), sm=state_machine)
    second = atom_runtime_factory(ib=ibi.IB(), sm=state_machine)

    assert first is not second
    assert Atom().ib is second.ib


def test_harness_can_install_on_custom_atom_class(monkeypatch, state_machine) -> None:
    class LocalAtom(Atom):
        pass

    runtime = AtomRuntimeHarness(
        ib=ibi.IB(),
        sm=state_machine,
        contract_registry=ContractRegistry(),
    )

    runtime.install(monkeypatch, LocalAtom)

    assert LocalAtom().ib is runtime.ib
