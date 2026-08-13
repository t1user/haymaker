from dataclasses import dataclass
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.validators import (
    Validator,
    aware_datetime,
    finite_number,
    ib_contract,
    non_empty_string,
    optional_aware_datetime,
    qualified_contract,
    readonly_mapping,
)


def is_int(value):
    if not isinstance(value, int):
        raise ValueError("Value must be an int.")


def test_validator_correct_value():
    class A:
        a = Validator(is_int)

        def __init__(self, a):
            self.a = a

    aaa = A(5)
    assert aaa.a == 5


def test_validator_incorrect_value():
    class A:
        a = Validator(is_int)

        def __init__(self, a):
            self.a = a

    with pytest.raises(ValueError):
        A("a")


def test_validator_correct_value_with_dataclass():
    @dataclass
    class A:
        a: int  # type: ignore

    class B(A):
        a = Validator(is_int)

    aaa = B(5)
    assert aaa.a == 5


def test_validator_incorrect_value_with_dataclass():
    @dataclass
    class A:
        a: int  # type: ignore

    class B(A):
        a = Validator(is_int)

    with pytest.raises(ValueError):
        B("xxx")


def test_validator_correct_value_with_inheriting_dataclass():
    class B:
        a = Validator(is_int)

    @dataclass
    class A(B):
        a: int  # type: ignore

    aaa = A(5)
    assert aaa.a == 5


def test_validator_incorrect_value_with_inheriting_dataclass():
    class B:
        a = Validator(is_int)

    @dataclass
    class A(B):
        a: int  # type: ignore

    with pytest.raises(ValueError):
        A("xxx")


def test_aware_datetime_distinguishes_type_and_timezone_errors():
    aware = datetime.now(timezone.utc)

    assert aware_datetime(aware, "created_at") is aware
    assert optional_aware_datetime(None, "as_of") is None
    with pytest.raises(TypeError, match="created_at must be a datetime"):
        aware_datetime(None, "created_at")
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        aware_datetime(datetime(2026, 1, 1), "created_at")


def test_finite_number_normalizes_reals_and_rejects_bool_and_non_finite():
    assert finite_number(2, "quantity") == 2.0
    with pytest.raises(TypeError, match="quantity must be a real number"):
        finite_number(True, "quantity")
    with pytest.raises(ValueError, match="quantity must be finite"):
        finite_number(float("nan"), "quantity")


def test_readonly_mapping_copies_only_the_top_level():
    nested = []
    original = {"nested": nested}

    copied = readonly_mapping(original, "metadata")
    original["new"] = 1
    nested.append(2)

    assert "new" not in copied
    assert copied["nested"] == [2]
    with pytest.raises(TypeError):
        copied["new"] = 3


def test_non_empty_string_distinguishes_type_and_value_errors():
    assert non_empty_string("alpha", "source_key") == "alpha"
    with pytest.raises(TypeError, match="source_key must be a string"):
        non_empty_string(1, "source_key")
    with pytest.raises(ValueError, match="source_key must not be empty"):
        non_empty_string("", "source_key")


def test_contract_validation_distinguishes_blueprints_from_concrete_contracts():
    blueprint = ibi.Future(symbol="ES")
    concrete = ibi.Future(conId=1, symbol="ES")

    assert ib_contract(blueprint) is blueprint
    assert qualified_contract(concrete) is concrete
    with pytest.raises(TypeError, match="ib_insync.Contract"):
        ib_contract("ES")
    with pytest.raises(ValueError, match="non-zero conId"):
        qualified_contract(blueprint)
