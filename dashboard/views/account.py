"""IB account Streamlit view."""

from __future__ import annotations

import streamlit as st

from dashboard.config import DashboardConfig
from dashboard.data.account import (
    account_values_frame,
    client_id,
    connect_ib,
    open_position_pnl_frame,
)
from dashboard.views.formatting import round_money


@st.cache_resource(show_spinner=False)
def _ib_connection(host: str, port: int, client_id: int):
    return connect_ib(host=host, port=port, client_id=client_id)


def _read_account_data(ib, account: str, *, force_refresh: bool = False):
    if not ib.isConnected():
        raise ConnectionError("Interactive Brokers connection is not active.")
    return account_values_frame(ib), open_position_pnl_frame(
        ib, account, force_refresh=force_refresh
    )


def _connect(config: DashboardConfig):
    return _ib_connection(config.ib_host, config.ib_port, config.ib_client_id)


def _connect_and_read(config: DashboardConfig, *, force_refresh: bool = False):
    ib = _connect(config)
    return ib, *_read_account_data(
        ib, config.ib_account, force_refresh=force_refresh
    )


def render(config: DashboardConfig) -> None:
    header, refresh = st.columns([1, 1])
    header.header("Account")
    force_refresh = refresh.button("Refresh account", key="refresh_account")

    try:
        ib, account_values, position_pnl = _connect_and_read(
            config, force_refresh=force_refresh
        )
    except Exception as exc:
        try:
            _ib_connection.clear()
            ib, account_values, position_pnl = _connect_and_read(
                config, force_refresh=force_refresh
            )
            st.caption(
                "The cached IB connection could not provide account data, "
                "so a new connection was created."
            )
        except Exception as reconnect_exc:
            st.error(
                "Could not read Interactive Brokers account data. Start or "
                "reconnect the gateway/TWS first. "
                f"Initial error: {exc}. Reconnect error: {reconnect_exc}"
            )
            return

    st.caption(f"Connected to IB with clientId {client_id(ib)}")

    st.subheader("Account values")
    st.dataframe(round_money(account_values), width="stretch")

    st.subheader("Open-position PnL")
    st.dataframe(round_money(position_pnl), width="stretch")
