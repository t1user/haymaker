"""Blotter Streamlit view."""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import pandas as pd
import streamlit as st
from pymongo.database import Database  # type: ignore

from dashboard.config import DashboardConfig
from dashboard.data import blotter
from dashboard.data.orders import contract_multiplier_map
from dashboard.views.formatting import round_money


LOOKBACKS = {
    "All": None,
    "Today": 0,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Year to date": "ytd",
    "Custom": "custom",
}


def _utc_bounds(label: str) -> tuple[dt.datetime | None, dt.datetime | None]:
    now = dt.datetime.now(dt.timezone.utc)
    start: dt.datetime | None
    end: dt.datetime | None = now + dt.timedelta(seconds=1)
    value = LOOKBACKS[label]
    if value is None:
        start = None
        end = None
    elif value == "ytd":
        start = dt.datetime(now.year, 1, 1, tzinfo=dt.timezone.utc)
    elif value == "custom":
        start = now - dt.timedelta(days=30)
    elif value == 0:
        start = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
    else:
        start = now - dt.timedelta(days=int(value))
    return start, end


def _custom_bounds() -> tuple[dt.datetime, dt.datetime]:
    default_start = dt.date.today() - dt.timedelta(days=30)
    default_end = dt.date.today()
    start_date, end_date = cast(
        tuple[dt.date, dt.date],
        st.date_input(
            "Custom lookback",
            value=(default_start, default_end),
            key="blotter_custom_lookback",
        ),
    )
    start = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
    end = dt.datetime.combine(
        end_date + dt.timedelta(days=1),
        dt.time.min,
        tzinfo=dt.timezone.utc,
    )
    return start, end


def _display_money_metrics(positions: pd.DataFrame) -> None:
    if positions.empty:
        st.info("No positions found for the selected period.")
        return
    cols = st.columns(5)
    cols[0].metric("Closed positions", f"{len(positions):,}")
    cols[1].metric("PnL", f"{positions['pnl'].sum():,.2f}")
    cols[2].metric("Commissions", f"{positions['commissions'].sum():,.2f}")
    cols[3].metric("Win rate", f"{blotter.win_rate(positions['pnl']):.1%}")
    cols[4].metric("Payout ratio", f"{blotter.payout_ratio(positions['pnl']):.2f}")


def render(db: Database, config: DashboardConfig) -> None:
    header, refresh = st.columns([1, 1])
    header.header("Blotter")
    refresh.button("Refresh blotter", key="refresh_blotter")

    controls = st.container(horizontal=True)
    with controls:
        lookback = st.selectbox("Lookback", list(LOOKBACKS), index=0)
        recent_positions = st.number_input(
            "Recent positions",
            min_value=5,
            max_value=500,
            value=50,
            step=5,
        )
        recent_transactions = st.number_input(
            "Recent transactions",
            min_value=5,
            max_value=1000,
            value=100,
            step=10,
        )

    start, end = (
        _custom_bounds() if lookback == "Custom" else _utc_bounds(lookback)
    )
    orders_collection = db[config.orders_collection]
    multiplier_map = contract_multiplier_map(orders_collection)

    all_docs = blotter.fetch_blotter_docs(db[config.blotter_collection])
    all_transactions = blotter.transactions_frame(
        all_docs, multiplier_map=multiplier_map
    )
    all_positions = blotter.positions_frame(all_transactions)
    audit_positions = blotter.audit_positions_frame(all_transactions)
    positions = blotter.filter_positions_by_close_time(
        all_positions, start=start, end=end
    )
    recent_tx_docs = blotter.fetch_blotter_docs(
        db[config.blotter_collection], limit=int(recent_transactions)
    )
    recent_transactions_frame = blotter.transactions_frame(
        recent_tx_docs, multiplier_map=multiplier_map
    )

    _display_money_metrics(positions)

    st.subheader("Blotter by position")
    display = round_money(
        blotter.position_display_frame(positions).head(int(recent_positions))
    )
    selection: Any = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="positions_table",
    )
    selected_position_id = None
    selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
    if selected_rows:
        selected = display.iloc[int(selected_rows[0])]
        selected_position_id = selected.get("position_id")
        st.caption(f"Selected position_id: {selected_position_id}")
    manual_position_id = st.text_input("Raw documents for position_id")
    position_id = selected_position_id or manual_position_id
    if position_id:
        docs = blotter.fetch_position_docs(
            db[config.blotter_collection], position_id
        )
        st.json(blotter.strip_mongo_ids(docs), expanded=False)

    st.subheader("Blotter by transaction")
    st.dataframe(
        round_money(blotter.transaction_display_frame(recent_transactions_frame)),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Stats by strategy")
    st.dataframe(round_money(blotter.strategy_stats(positions)), width="stretch")

    stats_cols = st.columns(2)
    with stats_cols[0]:
        st.subheader("PnL by contract")
        st.dataframe(round_money(blotter.pnl_by_contract(positions)), width="stretch")
    with stats_cols[1]:
        st.subheader("PnL by month")
        st.dataframe(round_money(blotter.pnl_by_month(positions)), width="stretch")

    st.subheader("Roll audit")
    st.dataframe(
        round_money(blotter.rolls_frame(all_transactions)),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Data issues")
    st.caption(
        "Issues are rows that cannot be safely included in closed-position PnL "
        "or require attention: missing position IDs, missing multipliers, "
        "positions without exactly one OPEN row, open/unmatched positions, "
        "or multiple close rows."
    )
    st.dataframe(
        blotter.issues_frame(all_transactions, audit_positions),
        width="stretch",
        hide_index=True,
    )
