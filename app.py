import json
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Input, Output, State, callback_context, dcc, html

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INVESTMENTS = ["VIXL", "REAL", "JPM", "MSFT"]
DEFAULT_WATCHLIST = ["DFEU", "RNMBY"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_data(tickers, period="6mo"):
    """Download adjusted-close data for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # yf.download returns multi-level columns when >1 ticker
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            # Single ticker – rename column
            df = df[["Close"]].rename(columns={"Close": tickers[0]})
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def compute_returns(prices):
    """Return dict of DataFrames: daily, weekly, monthly returns (%)."""
    if prices.empty:
        return {}, {}, {}
    daily = prices.pct_change().iloc[1:] * 100
    weekly = prices.resample("W-FRI").last().pct_change().iloc[1:] * 100
    monthly = prices.resample("ME").last().pct_change().iloc[1:] * 100
    return daily, weekly, monthly


def make_heatmap(returns_df, title, last_n=None):
    """Build a Plotly heatmap figure from a returns DataFrame."""
    if returns_df is None or returns_df.empty:
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

    # Format dates
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


def make_summary_table(prices):
    """Return a summary table with latest price and period returns."""
    if prices is None or prices.empty:
        return html.Div("No data available", className="text-muted p-3")

    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        latest = s.iloc[-1]
        day_ret = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100 if len(s) >= 2 else 0
        week_ago = s.index[-1] - timedelta(days=7)
        week_vals = s[s.index >= week_ago]
        week_ret = ((week_vals.iloc[-1] / week_vals.iloc[0]) - 1) * 100 if len(week_vals) >= 2 else 0
        month_ago = s.index[-1] - timedelta(days=30)
        month_vals = s[s.index >= month_ago]
        month_ret = ((month_vals.iloc[-1] / month_vals.iloc[0]) - 1) * 100 if len(month_vals) >= 2 else 0

        def color_cell(val):
            color = "#2e7d32" if val >= 0 else "#d32f2f"
            return html.Td(f"{val:+.2f}%", style={"color": color, "fontWeight": "600"})

        rows.append(
            html.Tr([
                html.Td(ticker, style={"fontWeight": "700"}),
                html.Td(f"${latest:.2f}"),
                color_cell(day_ret),
                color_cell(week_ret),
                color_cell(month_ret),
            ])
        )

    if not rows:
        return html.Div("No data available", className="text-muted p-3")

    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Ticker"),
                    html.Th("Price"),
                    html.Th("1D"),
                    html.Th("1W"),
                    html.Th("1M"),
                ])
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        size="sm",
        className="mb-0",
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Stock & ETF Dashboard",
)

app.layout = dbc.Container(
    fluid=True,
    className="py-3",
    children=[
        # Stores for ticker lists
        dcc.Store(id="investments-store", data=DEFAULT_INVESTMENTS),
        dcc.Store(id="watchlist-store", data=DEFAULT_WATCHLIST),
        # Header
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Stock & ETF Performance Dashboard",
                    className="text-center my-3",
                ),
            )
        ),
        # Time-range selector
        dbc.Row(
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button("1 Month", id="btn-1mo", n_clicks=0, outline=True, color="primary"),
                        dbc.Button("3 Months", id="btn-3mo", n_clicks=0, outline=True, color="primary"),
                        dbc.Button("6 Months", id="btn-6mo", n_clicks=0, color="primary"),
                        dbc.Button("1 Year", id="btn-1y", n_clicks=0, outline=True, color="primary"),
                    ],
                    className="mb-3",
                ),
                width="auto",
                className="d-flex justify-content-center",
            )
        ),
        dcc.Store(id="period-store", data="6mo"),
        # Two-panel layout
        dbc.Row(
            [
                # ---- Current Investments ----
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Current Investments", className="mb-0")),
                            dbc.CardBody(
                                [
                                    # Ticker management
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="inv-input",
                                                placeholder="Add ticker (e.g. AAPL)",
                                                type="text",
                                            ),
                                            dbc.Button("Add", id="inv-add-btn", color="success", n_clicks=0),
                                        ],
                                        className="mb-2",
                                        size="sm",
                                    ),
                                    html.Div(id="inv-badges", className="mb-3"),
                                    # Summary table
                                    html.Div(id="inv-summary"),
                                    # Chart
                                    dcc.Loading(dcc.Graph(id="inv-price-chart")),
                                    # Heatmap tabs
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="inv-heatmap-daily")),
                                                label="Daily",
                                            ),
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="inv-heatmap-weekly")),
                                                label="Weekly",
                                            ),
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="inv-heatmap-monthly")),
                                                label="Monthly",
                                            ),
                                        ],
                                        className="mt-3",
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
                # ---- Watchlist ----
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Watchlist", className="mb-0")),
                            dbc.CardBody(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="watch-input",
                                                placeholder="Add ticker (e.g. TSLA)",
                                                type="text",
                                            ),
                                            dbc.Button("Add", id="watch-add-btn", color="success", n_clicks=0),
                                        ],
                                        className="mb-2",
                                        size="sm",
                                    ),
                                    html.Div(id="watch-badges", className="mb-3"),
                                    html.Div(id="watch-summary"),
                                    dcc.Loading(dcc.Graph(id="watch-price-chart")),
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="watch-heatmap-daily")),
                                                label="Daily",
                                            ),
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="watch-heatmap-weekly")),
                                                label="Weekly",
                                            ),
                                            dbc.Tab(
                                                dcc.Loading(dcc.Graph(id="watch-heatmap-monthly")),
                                                label="Monthly",
                                            ),
                                        ],
                                        className="mt-3",
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
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

# Period selector
@app.callback(
    Output("period-store", "data"),
    Output("btn-1mo", "outline"),
    Output("btn-3mo", "outline"),
    Output("btn-6mo", "outline"),
    Output("btn-1y", "outline"),
    Input("btn-1mo", "n_clicks"),
    Input("btn-3mo", "n_clicks"),
    Input("btn-6mo", "n_clicks"),
    Input("btn-1y", "n_clicks"),
    prevent_initial_call=True,
)
def update_period(*_):
    btn = callback_context.triggered_id
    mapping = {"btn-1mo": "1mo", "btn-3mo": "3mo", "btn-6mo": "6mo", "btn-1y": "1y"}
    period = mapping.get(btn, "6mo")
    outlines = [btn != b for b in mapping]
    return (period, *outlines)


# --- Investment ticker management ---
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
    if triggered == "inv-add-btn":
        if new_ticker:
            ticker = new_ticker.strip().upper()
            if ticker and ticker not in current:
                return current + [ticker]
    elif isinstance(triggered, dict) and triggered.get("type") == "inv-remove":
        ticker = triggered["ticker"]
        return [t for t in current if t != ticker]
    return current


@app.callback(
    Output("watch-store", "data") if False else Output("watchlist-store", "data"),
    Input("watch-add-btn", "n_clicks"),
    Input({"type": "watch-remove", "ticker": dash.ALL}, "n_clicks"),
    State("watch-input", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=True,
)
def manage_watchlist(add_clicks, remove_clicks, new_ticker, current):
    triggered = callback_context.triggered_id
    if triggered == "watch-add-btn":
        if new_ticker:
            ticker = new_ticker.strip().upper()
            if ticker and ticker not in current:
                return current + [ticker]
    elif isinstance(triggered, dict) and triggered.get("type") == "watch-remove":
        ticker = triggered["ticker"]
        return [t for t in current if t != ticker]
    return current


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
                        style={"cursor": "pointer", "marginLeft": "4px", "fontSize": "1rem"},
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


# Clear input after adding
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


# --- Main data callbacks ---
@app.callback(
    Output("inv-summary", "children"),
    Output("inv-price-chart", "figure"),
    Output("inv-heatmap-daily", "figure"),
    Output("inv-heatmap-weekly", "figure"),
    Output("inv-heatmap-monthly", "figure"),
    Input("investments-store", "data"),
    Input("period-store", "data"),
)
def update_investments(tickers, period):
    prices = fetch_data(tickers, period)
    daily, weekly, monthly = compute_returns(prices)
    return (
        make_summary_table(prices),
        make_price_chart(prices, "Investment Performance (Indexed)"),
        make_heatmap(daily, "Daily Returns (%)", last_n=30),
        make_heatmap(weekly, "Weekly Returns (%)"),
        make_heatmap(monthly, "Monthly Returns (%)"),
    )


@app.callback(
    Output("watch-summary", "children"),
    Output("watch-price-chart", "figure"),
    Output("watch-heatmap-daily", "figure"),
    Output("watch-heatmap-weekly", "figure"),
    Output("watch-heatmap-monthly", "figure"),
    Input("watchlist-store", "data"),
    Input("period-store", "data"),
)
def update_watchlist(tickers, period):
    prices = fetch_data(tickers, period)
    daily, weekly, monthly = compute_returns(prices)
    return (
        make_summary_table(prices),
        make_price_chart(prices, "Watchlist Performance (Indexed)"),
        make_heatmap(daily, "Daily Returns (%)", last_n=30),
        make_heatmap(weekly, "Weekly Returns (%)"),
        make_heatmap(monthly, "Monthly Returns (%)"),
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
