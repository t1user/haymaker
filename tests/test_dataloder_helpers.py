from datetime import timedelta

import pytest

from haymaker.dataloader.helpers import (
    duration_in_secs,
    duration_str,
    timedelta_and_barSize_to_duration_str,
    timedelta_normalizer,
    timedelta_to_duration_in_secs,
)


def test_duration_in_secs_1_hour():
    assert duration_in_secs("1 hours") == 3600


def test_duration_in_secs_1_hour_drop_s():
    assert duration_in_secs("1 hour") == 3600


def test_duration_in_secs():
    assert duration_in_secs("10 mins") == 600


def test_duration_str():
    assert duration_str(3600) == "3600 S"


@pytest.mark.parametrize(
    "duration, bar_size, output",
    [
        (timedelta(days=1), 3600, timedelta_normalizer(24 * 60 * 60)),
        (timedelta(days=4 * 30), 30, 3_000_000),
        (timedelta(hours=1), 60, 3600),
    ],
)
def test_timedelta_and_barSize_to_duration_in_secs(duration, bar_size, output):
    assert timedelta_to_duration_in_secs(duration, bar_size, max_bars=100_000) == output


@pytest.mark.parametrize(
    "duration, barSize, output",
    [
        (timedelta(hours=1), "1 hour", "3600 S"),
        (timedelta(days=4 * 30), "30 secs", "1 M"),
        (timedelta(hours=1), "1 min", "3600 S"),
    ],
)
def test_timedelta_and_barSize_to_duration_str(duration, barSize, output):
    assert (
        timedelta_and_barSize_to_duration_str(duration, barSize, max_bars=100_000)
        == output
    )


def test_timedelta_and_barSize_to_duration_str_secs_fixed():
    assert timedelta_and_barSize_to_duration_str(timedelta(days=5), "1 sec") == "2000 S"


def test_timedelta_normalizer():
    four_months = timedelta(days=4 * 30).total_seconds()
    assert timedelta_normalizer(four_months) < four_months


def test_timedelta_normalizer_1():
    one_hour = timedelta(hours=1).total_seconds()
    assert timedelta_normalizer(one_hour) == one_hour


def test_timedelta_normalizer_2():
    one_working_week = timedelta(days=5).total_seconds()
    assert timedelta_normalizer(one_working_week) == 5 * 23 * 60 * 60


def test_timedelta_normalizer_3():
    one_working_week = timedelta(days=5).total_seconds()
    one_regular_week = timedelta(days=7).total_seconds()
    assert timedelta_normalizer(one_regular_week) == timedelta_normalizer(
        one_working_week
    )


def test_timedelta_normalizer_4():
    two_weeks = timedelta(weeks=2).total_seconds()
    assert timedelta_normalizer(two_weeks) == 10 * 23 * 60 * 60
