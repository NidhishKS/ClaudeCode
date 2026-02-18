import io
import json
from datetime import timedelta
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Input, Output, State, callback_context, dcc, html

# ---------------------------------------------------------------------------
# Defaults & persistence
# ---------------------------------------------------------------------------
DEFAULT_INVESTMENTS = ["VIXL", "REAL", "JPM", "MSFT"]
DEFAULT_WATCHLIST = ["DFEU", "RNMBY"]

TICKERS_FILE = Path(__file__).parent / "tickers.json"


def load_tickers():
    """Load saved ticker lists from disk, falling back to defaults."""
    if TICKERS_FILE.exists():
        try:
            with open(TICKERS_FILE, "r") as f:
                data = json.load(f)
            return (
                data.get("investments", DEFAULT_INVESTMENTS),
                data.get("watchlist", DEFAULT_WATCHLIST),
            )
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_INVESTMENTS[:], DEFAULT_WATCHLIST[:]


def save_tickers(investments, watchlist):
    """Persist current ticker lists to disk."""
    with open(TICKERS_FILE, "w") as f:
        json.dump({"investments": investments, "watchlist": watchlist}, f, indent=2)


SAVED_INVESTMENTS, SAVED_WATCHLIST = load_tickers()

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def fetch_data(tickers, period="1y"):
    """Download adjusted-close data for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            df = df[["Close"]].rename(columns={"Close": tickers[0]})
        # Only keep the exact tickers requested (prevents cross-contamination)
        valid_cols = [t for t in tickers if t in df.columns]
        df = df[valid_cols]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def slice_by_period(prices, period):
    """Slice a prices DataFrame to the given period from the end."""
    if prices.empty:
        return prices
    end = prices.index[-1]
    period_map = {
        "1w": timedelta(days=7),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
    }
    delta = period_map.get(period, timedelta(days=180))
    start = end - delta
    return prices[prices.index >= start]


def restore_prices(data_json):
    """Restore a prices DataFrame from its JSON representation."""
    if not data_json:
        return pd.DataFrame()
    prices = pd.read_json(io.StringIO(data_json), orient="split")
    prices.index = pd.to_datetime(prices.index)
    return prices


def compute_returns(prices):
    """Return DataFrames: daily, weekly, monthly returns (%)."""
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    daily = prices.pct_change().iloc[1:] * 100
    weekly = prices.resample("W-FRI").last().pct_change().iloc[1:] * 100
    monthly = prices.resample("ME").last().pct_change().iloc[1:] * 100
    return daily, weekly, monthly


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def make_heatmap(returns_df, title, last_n=7):
    """Build a Plotly heatmap figure from a returns DataFrame."""
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[
                dict(text="No data available", showarrow=False, font=dict(size=16))
            ],
        )
        return fig

    df = returns_df.copy()
    if last_n and len(df) > last_n:
        df = df.iloc[-last_n:]

    date_labels = [d.strftime("%Y-%m-%d") for d in df.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values.T,
            x=date_labels,
            y=list(df.columns),
            colorscale=[
                [0, "#d32f2f"],
                [0.5, "#ffffff"],
                [1, "#2e7d32"],
            ],
            zmid=0,
            text=np.round(df.values.T, 2),
            texttemplate="%{text:.1f}%",
            textfont={"size": 10},
            hovertemplate="Date: %{x}<br>Ticker: %{y}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=max(250, 60 * len(df.columns) + 100),
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return fig


def make_price_chart(prices, title):
    """Build a normalized price chart (base 100)."""
    if prices is None or prices.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[
                dict(text="No data available", showarrow=False, font=dict(size=16))
            ],
        )
        return fig

    normalized = (prices / prices.iloc[0]) * 100
    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + col + "</extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Indexed Price (base = 100)",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def make_individual_chart(prices, ticker):
    """Build a raw (non-indexed) price chart for a single ticker."""
    if prices is None or prices.empty or ticker not in prices.columns:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(text="Select a ticker above", showarrow=False, font=dict(size=14))
            ],
            height=400,
        )
        return fig

    s = prices[ticker].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=ticker))
    fig.update_layout(
        title=f"{ticker} Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _calc_return(series, days):
    """Calculate return over approximately `days` calendar days."""
    cutoff = series.index[-1] - timedelta(days=days)
    vals = series[series.index >= cutoff]
    if len(vals) >= 2:
        return ((vals.iloc[-1] / vals.iloc[0]) - 1) * 100
    return 0.0


def make_summary_table(prices):
    """Return a summary table with latest price and period returns."""
    if prices is None or prices.empty:
        return html.Div("No data available", className="text-muted p-3")

    def color_cell(val):
        color = "#2e7d32" if val >= 0 else "#d32f2f"
        return html.Td(f"{val:+.2f}%", style={"color": color, "fontWeight": "600"})

    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        latest = s.iloc[-1]
        day_ret = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100
        week_ret = _calc_return(s, 7)
        month_ret = _calc_return(s, 30)
        three_month_ret = _calc_return(s, 90)
        six_month_ret = _calc_return(s, 180)
        year_ret = _calc_return(s, 365)

        rows.append(
            html.Tr(
                [
                    html.Td(ticker, style={"fontWeight": "700"}),
                    html.Td(f"${latest:.2f}"),
                    color_cell(day_ret),
                    color_cell(week_ret),
                    color_cell(month_ret),
                    color_cell(three_month_ret),
                    color_cell(six_month_ret),
                    color_cell(year_ret),
                ]
            )
        )

    if not rows:
        return html.Div("No data available", className="text-muted p-3")

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Ticker"),
                        html.Th("Price"),
                        html.Th("1D"),
                        html.Th("1W"),
                        html.Th("1M"),
                        html.Th("3M"),
                        html.Th("6M"),
                        html.Th("1Y"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        size="sm",
        className="mb-0",
        responsive=True,
    )


# ---------------------------------------------------------------------------
# App & layout
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Stock & ETF Dashboard",
)


def _period_btn_group(prefix, color="primary"):
    """Create a 1W / 1M / 3M / 6M / 1Y button group for charts."""
    periods = [("1W", "1w"), ("1M", "1mo"), ("3M", "3mo"), ("6M", "6mo"), ("1Y", "1y")]
    buttons = []
    for label, key in periods:
        is_default = key == "6mo"
        buttons.append(
            dbc.Button(
                label,
                id=f"{prefix}-btn-{key}",
                n_clicks=0,
                outline=not is_default,
                color=color,
                size="sm",
            )
        )
    return dbc.ButtonGroup(buttons, className="mb-2")


def make_panel(prefix, title):
    """Create the layout for one panel (investments or watchlist)."""
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(title, className="mb-0")),
            dbc.CardBody(
                [
                    # Ticker management
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id=f"{prefix}-input",
                                placeholder="Add ticker (e.g. AAPL)",
                                type="text",
                            ),
                            dbc.Button(
                                "Add",
                                id=f"{prefix}-add-btn",
                                color="success",
                                n_clicks=0,
                            ),
                        ],
                        className="mb-2",
                        size="sm",
                    ),
                    html.Div(id=f"{prefix}-badges", className="mb-3"),
                    # Summary table
                    html.Div(id=f"{prefix}-summary"),
                    # Portfolio performance chart
                    html.H6("Portfolio Performance", className="mt-3 mb-2"),
                    _period_btn_group(f"{prefix}-chart", color="primary"),
                    dcc.Store(id=f"{prefix}-chart-period", data="6mo"),
                    dcc.Loading(dcc.Graph(id=f"{prefix}-price-chart")),
                    # Individual stock chart
                    html.H6("Individual Stock", className="mt-3 mb-2"),
                    dcc.Dropdown(
                        id=f"{prefix}-ind-dropdown",
                        placeholder="Select a ticker...",
                        className="mb-2",
                    ),
                    _period_btn_group(f"{prefix}-ind", color="secondary"),
                    dcc.Store(id=f"{prefix}-ind-period", data="6mo"),
                    dcc.Loading(dcc.Graph(id=f"{prefix}-individual-chart")),
                    # Heatmaps
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                dcc.Loading(dcc.Graph(id=f"{prefix}-heatmap-daily")),
                                label="Daily",
                            ),
                            dbc.Tab(
                                dcc.Loading(dcc.Graph(id=f"{prefix}-heatmap-weekly")),
                                label="Weekly",
                            ),
                            dbc.Tab(
                                dcc.Loading(dcc.Graph(id=f"{prefix}-heatmap-monthly")),
                                label="Monthly",
                            ),
                        ],
                        className="mt-3",
                    ),
                ]
            ),
        ],
        className="shadow-sm",
    )


app.layout = dbc.Container(
    fluid=True,
    className="py-3",
    children=[
        # Stores for ticker lists & cached price data
        dcc.Store(id="investments-store", data=SAVED_INVESTMENTS),
        dcc.Store(id="watchlist-store", data=SAVED_WATCHLIST),
        dcc.Store(id="inv-data-store"),
        dcc.Store(id="watch-data-store"),
        # Header
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Stock & ETF Performance Dashboard",
                    className="text-center my-3",
                ),
            )
        ),
        # Two-panel layout
        dbc.Row(
            [
                dbc.Col(make_panel("inv", "Current Investments"), lg=6),
                dbc.Col(make_panel("watch", "Watchlist"), lg=6),
            ]
        ),
        # Footer
        dbc.Row(
            dbc.Col(
                html.P(
                    "Data provided by Yahoo Finance. Prices may be delayed.",
                    className="text-muted text-center mt-4 small",
                )
            )
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# --- Ticker management ---


@app.callback(
    Output("investments-store", "data"),
    Input("inv-add-btn", "n_clicks"),
    Input({"type": "inv-remove", "ticker": dash.ALL}, "n_clicks"),
    State("inv-input", "value"),
    State("investments-store", "data"),
    prevent_initial_call=True,
)
def manage_investments(add_clicks, remove_clicks, new_ticker, current):
    triggered = callback_context.triggered_id
    updated = current
    if triggered == "inv-add-btn":
        if new_ticker:
            ticker = new_ticker.strip().upper()
            if ticker and ticker not in current:
                updated = current + [ticker]
    elif isinstance(triggered, dict) and triggered.get("type") == "inv-remove":
        ticker = triggered["ticker"]
        updated = [t for t in current if t != ticker]
    if updated is not current:
        save_tickers(updated, load_tickers()[1])
    return updated


@app.callback(
    Output("watchlist-store", "data"),
    Input("watch-add-btn", "n_clicks"),
    Input({"type": "watch-remove", "ticker": dash.ALL}, "n_clicks"),
    State("watch-input", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=True,
)
def manage_watchlist(add_clicks, remove_clicks, new_ticker, current):
    triggered = callback_context.triggered_id
    updated = current
    if triggered == "watch-add-btn":
        if new_ticker:
            ticker = new_ticker.strip().upper()
            if ticker and ticker not in current:
                updated = current + [ticker]
    elif isinstance(triggered, dict) and triggered.get("type") == "watch-remove":
        ticker = triggered["ticker"]
        updated = [t for t in current if t != ticker]
    if updated is not current:
        save_tickers(load_tickers()[0], updated)
    return updated


# --- Render ticker badges ---


def render_badges(tickers, badge_type):
    badges = []
    for t in tickers:
        badges.append(
            dbc.Badge(
                [
                    t + " ",
                    html.Span(
                        "\u00d7",
                        id={"type": f"{badge_type}-remove", "ticker": t},
                        n_clicks=0,
                        style={
                            "cursor": "pointer",
                            "marginLeft": "4px",
                            "fontSize": "1rem",
                        },
                    ),
                ],
                color="primary",
                className="me-1 mb-1 p-2",
            )
        )
    return badges


@app.callback(Output("inv-badges", "children"), Input("investments-store", "data"))
def show_inv_badges(tickers):
    return render_badges(tickers, "inv")


@app.callback(Output("watch-badges", "children"), Input("watchlist-store", "data"))
def show_watch_badges(tickers):
    return render_badges(tickers, "watch")


# --- Clear input after adding ---


@app.callback(
    Output("inv-input", "value"),
    Input("inv-add-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_inv_input(_):
    return ""


@app.callback(
    Output("watch-input", "value"),
    Input("watch-add-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_watch_input(_):
    return ""


# --- Data caching: fetch 1y data once when tickers change ---


@app.callback(Output("inv-data-store", "data"), Input("investments-store", "data"))
def cache_inv_data(tickers):
    prices = fetch_data(tickers, "1y")
    if prices.empty:
        return None
    return prices.to_json(date_format="iso", orient="split")


@app.callback(Output("watch-data-store", "data"), Input("watchlist-store", "data"))
def cache_watch_data(tickers):
    prices = fetch_data(tickers, "1y")
    if prices.empty:
        return None
    return prices.to_json(date_format="iso", orient="split")


# --- Panel-specific callbacks (registered via factory) ---


def register_panel_callbacks(prefix, data_store_id, chart_title, tickers_store_id):
    """Register all display callbacks for a panel."""

    chart_btn_ids = [f"{prefix}-chart-btn-{p}" for p in ["1w", "1mo", "3mo", "6mo", "1y"]]
    chart_periods = ["1w", "1mo", "3mo", "6mo", "1y"]

    # Chart period selector
    @app.callback(
        Output(f"{prefix}-chart-period", "data"),
        *[Output(bid, "outline") for bid in chart_btn_ids],
        *[Input(bid, "n_clicks") for bid in chart_btn_ids],
        prevent_initial_call=True,
    )
    def update_chart_period(*_, _ids=chart_btn_ids, _periods=chart_periods):
        btn = callback_context.triggered_id
        mapping = dict(zip(_ids, _periods))
        period = mapping.get(btn, "6mo")
        outlines = [btn != b for b in _ids]
        return (period, *outlines)

    # Individual chart period selector
    ind_btn_ids = [f"{prefix}-ind-btn-{p}" for p in ["1w", "1mo", "3mo", "6mo", "1y"]]

    @app.callback(
        Output(f"{prefix}-ind-period", "data"),
        *[Output(bid, "outline") for bid in ind_btn_ids],
        *[Input(bid, "n_clicks") for bid in ind_btn_ids],
        prevent_initial_call=True,
    )
    def update_ind_period(*_, _ids=ind_btn_ids, _periods=chart_periods):
        btn = callback_context.triggered_id
        mapping = dict(zip(_ids, _periods))
        period = mapping.get(btn, "6mo")
        outlines = [btn != b for b in _ids]
        return (period, *outlines)

    # Update dropdown options when tickers change
    @app.callback(
        Output(f"{prefix}-ind-dropdown", "options"),
        Input(tickers_store_id, "data"),
    )
    def update_dropdown(tickers):
        return [{"label": t, "value": t} for t in (tickers or [])]

    # Table + heatmaps (use full 1y cached data)
    @app.callback(
        Output(f"{prefix}-summary", "children"),
        Output(f"{prefix}-heatmap-daily", "figure"),
        Output(f"{prefix}-heatmap-weekly", "figure"),
        Output(f"{prefix}-heatmap-monthly", "figure"),
        Input(data_store_id, "data"),
    )
    def update_table_heatmaps(data_json):
        prices = restore_prices(data_json)
        daily, weekly, monthly = compute_returns(prices)
        return (
            make_summary_table(prices),
            make_heatmap(daily, "Daily Returns (%)", last_n=7),
            make_heatmap(weekly, "Weekly Returns (%)", last_n=7),
            make_heatmap(monthly, "Monthly Returns (%)", last_n=7),
        )

    # Portfolio performance chart (responds to its own period buttons)
    _chart_title = chart_title

    @app.callback(
        Output(f"{prefix}-price-chart", "figure"),
        Input(data_store_id, "data"),
        Input(f"{prefix}-chart-period", "data"),
    )
    def update_price_chart(data_json, period, _title=_chart_title):
        prices = restore_prices(data_json)
        if not prices.empty:
            prices = slice_by_period(prices, period)
        return make_price_chart(prices, _title)

    # Individual stock chart
    @app.callback(
        Output(f"{prefix}-individual-chart", "figure"),
        Input(data_store_id, "data"),
        Input(f"{prefix}-ind-dropdown", "value"),
        Input(f"{prefix}-ind-period", "data"),
    )
    def update_individual_chart(data_json, selected_ticker, period):
        if not selected_ticker:
            fig = go.Figure()
            fig.update_layout(
                annotations=[
                    dict(
                        text="Select a ticker above",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ],
                height=400,
            )
            return fig
        prices = restore_prices(data_json)
        if not prices.empty:
            prices = slice_by_period(prices, period)
        return make_individual_chart(prices, selected_ticker)


register_panel_callbacks(
    "inv", "inv-data-store", "Investment Performance (Indexed)", "investments-store"
)
register_panel_callbacks(
    "watch", "watch-data-store", "Watchlist Performance (Indexed)", "watchlist-store"
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
