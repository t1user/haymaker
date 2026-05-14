"""Shared normalization helpers for Mongo-backed dashboard tables."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


def ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def clean_object_id(value: Any) -> str:
    return str(value) if value is not None else ""


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def numeric(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def find_nested_value(obj: Any, key: str) -> Any:
    """Return the first scalar value under *key* in a nested document."""

    if isinstance(obj, dict):
        if key in obj and not isinstance(obj[key], (dict, list, tuple)):
            return obj[key]
        for value in obj.values():
            found = find_nested_value(value, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_nested_value(value, key)
            if found is not None:
                return found
    return None


def flatten_unique(values: Iterable[Any]) -> str:
    output = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        if value not in output:
            output.append(value)
    return ", ".join(map(str, output))
