"""Streamlit entrypoint for the local trading dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.config import get_config
from dashboard.data.mongo import get_database
from dashboard.views import account, blotter, orders


def main() -> None:
    st.set_page_config(
        page_title="Haymaker Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Haymaker Dashboard")

    config = get_config()
    try:
        db = get_database(config)
    except Exception as exc:
        st.error(f"Could not connect to MongoDB: {exc}")
        return

    tab_blotter, tab_orders, tab_account = st.tabs(
        ["Blotter", "Orders / Models", "Account"]
    )

    with tab_blotter:
        blotter.render(db, config)
    with tab_orders:
        orders.render(db, config)
    with tab_account:
        account.render(config)


if __name__ == "__main__":
    main()
