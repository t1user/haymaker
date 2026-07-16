"""Runtime fixtures for tests that exercise ``Atom`` services.

Haymaker production code installs a single ``RuntimeContext`` on ``Atom`` before
strategy objects are created. Tests should follow the same shape instead of
patching ``Atom.ib``, ``Atom.runtime``, or concrete ``contract_registry`` class
attributes directly.

Use ``atom_runtime`` when the default test runtime is enough. It installs a
fresh ``ib_insync.IB`` instance, the test ``StateMachine``, a
``ContractRegistry``, and a restart recorder on ``Atom``.

Use ``atom_runtime_factory`` when a test needs custom services:

    runtime = atom_runtime_factory(contract_registry=mock_registry)
    runtime.bind_controller(fake_controller)

Examples:

    def test_contract_registration(atom_runtime):
        class MyAtom(Atom):
            def __init__(self, contract):
                super().__init__()
                self.contract = contract

        contract = ibi.Future("ES", exchange="CME")
        MyAtom(contract)

        assert contract in atom_runtime.contract_registry.blueprints

    def test_signal_processor_uses_fake_state_machine(
        atom_runtime_factory, FakeStateMachine
    ):
        sm = FakeStateMachine(position=1)
        atom_runtime_factory(sm=sm)

        processor = BinarySignalProcessor()

        assert processor.position("strategy") == 1

    def test_execution_model_uses_fake_controller(atom_runtime):
        controller = FakeController(FakeTrader())
        atom_runtime.bind_controller(controller)

        model = BaseExecModel()

        assert model.controller is controller

The controller is intentionally opt-in. Many tests only need contract
registration, state-machine access, or restart recording; constructing a
controller wires IB events and should remain explicit in tests that need it.

When writing new tests:

- install runtime services through ``atom_runtime`` or
  ``atom_runtime_factory`` before creating contract-bearing Atoms;
- use ``atom_runtime.ib`` instead of ``Atom.ib``;
- use ``atom_runtime.contract_registry`` or a factory-supplied registry instead
  of assigning ``SomeAtom.contract_registry``;
- use ``runtime.bind_controller(controller)`` instead of monkeypatching
  ``Atom.runtime`` with a partial object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import ib_insync as ibi

from haymaker.base import Atom
from haymaker.config.settings import OrderDefaults, StorageSettings, TimeoutPolicy
from haymaker.contract_registry import ContractRegistry
from haymaker.controller import Controller
from haymaker.databases import StoreFactory
from haymaker.state_machine import StateMachine
from haymaker.trader import Trader


@dataclass
class AtomRuntimeHarness:
    """Test runtime context installed on ``Atom``."""

    ib: ibi.IB
    sm: StateMachine
    contract_registry: ContractRegistry
    controller: Controller | None = None
    trader: Trader | None = None
    store_factory: StoreFactory | None = None
    order_defaults: OrderDefaults = field(default_factory=OrderDefaults)
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    dataframe_save_frequency: int = 900
    restart_requests: list[str] = field(default_factory=list)
    future_roll_policies: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Create the default trader around the harness IB client."""

        if self.trader is None:
            self.trader = Trader(self.ib)
        if self.store_factory is None:
            self.store_factory = StoreFactory(StorageSettings())

    def install(self, monkeypatch: Any, atom_cls: type[Atom] = Atom) -> Self:
        """Install this harness on an Atom class for the current test."""

        monkeypatch.setattr(atom_cls, "runtime", self, raising=False)
        return self

    def bind_controller(self, controller: Controller) -> Controller:
        """Attach a controller to the installed runtime and return it."""

        self.controller = controller
        return controller

    def request_restart(self, reason: str = "") -> bool:
        """Record a restart request and report that it was accepted."""

        self.restart_requests.append(reason)
        return True
