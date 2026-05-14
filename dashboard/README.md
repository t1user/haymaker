# Haymaker Dashboard

Local Streamlit dashboard for trading state currently inspected through notebooks.

## Setup

Install the optional dashboard dependencies:

```bash
/home/tomek/.virtualenvs/new_ib/bin/python -m pip install -e ".[dashboard]"
```

Copy the local configuration template:

```bash
cp dashboard/.env.example dashboard/.env
```

Adjust Mongo collection names and IB connection settings in `dashboard/.env`.
The real `.env` file is ignored by git.

## Run

```bash
/home/tomek/.virtualenvs/new_ib/bin/python -m streamlit run dashboard/app.py
```

or, after installing the optional dependency group:

```bash
haymaker-dashboard
```

The account page reuses the notebook's direct IB connection approach. If the
gateway or TWS is not available, the page reports the connection error instead
of failing the whole dashboard.
