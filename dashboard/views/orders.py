"""Orders and model-state Streamlit view."""

from __future__ import annotations

import streamlit as st
from pymongo.database import Database  # type: ignore

from dashboard.config import DashboardConfig
from dashboard.data import orders
from dashboard.views.formatting import round_money


def render(db: Database, config: DashboardConfig) -> None:
    header, refresh = st.columns([1, 1])
    header.header("Orders / Models")
    refresh.button("Refresh orders / models", key="refresh_orders_models")
    collection = db[config.orders_collection]
    models_collection = db[config.models_collection]

    recent_count = st.number_input(
        "Recent orders",
        min_value=5,
        max_value=500,
        value=40,
        step=5,
    )

    recent_docs = orders.fetch_recent_order_docs(collection, int(recent_count))
    recent_orders = orders.orders_frame(recent_docs)
    active_docs = orders.fetch_active_order_docs(collection)
    active_orders = orders.orders_frame(active_docs)

    st.subheader("Recent orders")
    st.dataframe(round_money(recent_orders), width="stretch", hide_index=True)

    st.subheader("Open orders")
    st.dataframe(round_money(active_orders), width="stretch", hide_index=True)

    st.subheader("Order detail")
    order_id = st.number_input("orderId", min_value=0, step=1)
    if order_id:
        docs = orders.fetch_order_docs(collection, int(order_id))
        st.caption(f"Documents found: {len(docs)}")
        st.json(docs, expanded=False)

    st.subheader("Models")
    model_doc = orders.latest_model_doc(models_collection)
    st.dataframe(
        round_money(orders.models_frame(model_doc, active_orders)),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Raw latest model document"):
        st.json(model_doc or {}, expanded=False)
