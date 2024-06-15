from dataclasses import dataclass

import pytest

from ib_tools.validators import Validator


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
        a: int

    class B(A):
        a = Validator(is_int)

    aaa = B(5)
    assert aaa.a == 5


def test_validator_incorrect_value_with_dataclass():

    @dataclass
    class A:
        a: int

    class B(A):
        a = Validator(is_int)

    with pytest.raises(ValueError):
        B("xxx")


def test_validator_correct_value_with_inheriting_dataclass():

    class B:
        a = Validator(is_int)

    @dataclass
    class A(B):
        a: int

    aaa = A(5)
    assert aaa.a == 5


def test_validator_incorrect_value_with_inheriting_dataclass():

    class B:
        a = Validator(is_int)

    @dataclass
    class A(B):
        a: int

    with pytest.raises(ValueError):
        A("xxx")
