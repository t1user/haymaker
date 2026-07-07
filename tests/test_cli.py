import pytest

from haymaker.cli import load_user_module
from haymaker.runtime import RuntimeContext


def test_load_user_module_supports_sibling_imports(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 42\n")
    strategy = tmp_path / "strategy.py"
    strategy.write_text("from helper import VALUE\nresult = VALUE\n")

    module = load_user_module(strategy)

    assert module.result == 42


def test_read_no_future_roll_strategies_accepts_module_list(tmp_path):
    strategy = tmp_path / "strategy.py"
    strategy.write_text("no_future_roll_strategies = ['one', 'two']\n")

    module = load_user_module(strategy)

    assert RuntimeContext._read_no_future_roll_strategies(module) == ["one", "two"]


def test_read_no_future_roll_strategies_rejects_wrong_type(tmp_path):
    strategy = tmp_path / "strategy.py"
    strategy.write_text("no_future_roll_strategies = 'one'\n")

    module = load_user_module(strategy)

    with pytest.raises(TypeError):
        RuntimeContext._read_no_future_roll_strategies(module)
