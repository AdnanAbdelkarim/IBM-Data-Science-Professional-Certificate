# Works with either spacex.csv OR spacex.sqlite (table 'spacex')

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os, sqlite3

# ---------- Data loading ----------
def load_data():
    candidates = [
        "spacex.csv", "dataset.csv", "SpaceX.csv",
        "data/spacex.csv", "data/dataset.csv", "data/SpaceX.csv"
    ]
    for csv_name in candidates:
        if os.path.exists(csv_name):
            return pd.read_csv(csv_name)
    if os.path.exists("spacex.sqlite"):
        with sqlite3.connect("spacex.sqlite") as con:
            return pd.read_sql("SELECT * FROM spacex", con)
    if os.path.exists("data/spacex.sqlite"):
        with sqlite3.connect("data/spacex.sqlite") as con:
            return pd.read_sql("SELECT * FROM spacex", con)
    raise FileNotFoundError("Provide spacex.csv (preferred) or spacex.sqlite with table 'spacex'.")

df = load_data()

# ---------- Column normalization (handles your schema) ----------
def ensure_col(df, target, *cands):
    if target in df.columns:
        return target
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower_map:
            df[target] = df[lower_map[c.lower()]]
            return target
    if target == "class" and "Class" in df.columns:
        df["class"] = df["Class"]
        return "class"
    raise KeyError(f"Missing required column for '{target}'. Available: {list(df.columns)}")

ensure_col(df, "LaunchSite", "launch_site", "LaunchSite", "Launch Site")
ensure_col(df, "PayloadMass", "payload_mass_kg", "PayloadMass", "PAYLOADMASS__KG_", "Payload_Mass")
ensure_col(df, "Orbit", "orbit", "Orbit")
ensure_col(df, "class", "class", "Class", "landing_success")

# Parse year if available
if "date_utc" in df.columns or "Date" in df.columns:
    col_date = "date_utc" if "date_utc" in df.columns else "Date"
    df["_Year"] = pd.to_datetime(df[col_date], errors="coerce").dt.year

# ---------- Helpers for defaults ----------
site_stats = (
    df.groupby("LaunchSite")["class"]
      .agg(rate="mean", flights="count")
      .reset_index()
      .sort_values(["rate", "flights"], ascending=[False, False])
)

default_site = "ALL"
best_site = site_stats.iloc[0]["LaunchSite"] if len(site_stats) else None
payload_min = int(np.floor(df["PayloadMass"].min()))
payload_max = int(np.ceil(df["PayloadMass"].max()))

# Build non-overlapping numeric bin edges (e.g., 1000-kg bins)
bin_width = 1000
if payload_max <= payload_min + 1:
    sweet_lo, sweet_hi = payload_min, payload_max
    sweet_rate = float(df["class"].mean())
else:
    edges = np.arange(payload_min, payload_max + bin_width, bin_width, dtype=int)
    if edges[-1] < payload_max:
        edges = np.append(edges, payload_max)
    edges = np.unique(edges)

    d = df.loc[df["PayloadMass"].notna() & df["class"].notna()].copy()
    d["_bin"] = pd.cut(
        d["PayloadMass"],
        bins=edges,
        right=True,
        include_lowest=True
    )

    band_tbl = (
        d.groupby("_bin")["class"]
         .agg(rate="mean", flights="count")
         .reset_index()
    )

    band_tbl = band_tbl[band_tbl["flights"] >= 8]

    if band_tbl.empty:
        sweet_lo, sweet_hi = payload_min, payload_max
        sweet_rate = float(d["class"].mean())
    else:
        best = band_tbl.sort_values(["rate", "flights"], ascending=[False, False]).iloc[0]
        sweet_lo, sweet_hi = int(np.floor(best["_bin"].left)), int(np.ceil(best["_bin"].right))
        sweet_rate = float(best["rate"])

# ---------- App ----------
app = Dash(__name__)
app.title = "SpaceX Launch Analysis"

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"},
    children=[
        html.H2("SpaceX Launch Analysis & Prediction", style={"margin": "10px 0 0"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Launch Site", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="site-dropdown",
                            options=[{"label": "ALL", "value": "ALL"}]
                                    + [{"label": s, "value": s} for s in sorted(df["LaunchSite"].unique())],
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "220px", "marginRight": "12px"},
                ),
                html.Div(
                    [
                        html.Label("Payload Range (kg)", style={"fontWeight": "600"}),
                        dcc.RangeSlider(
                            id="payload-slider",
                            min=payload_min, max=payload_max, step=100,
                            value=[sweet_lo, sweet_hi],
                            tooltip={"always_visible": True}
                        ),
                    ],
                    style={"flex": "3"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "alignItems": "center", "margin": "10px 0 20px"},
        ),

        # KPI strip
        html.Div(
            [
                html.Div([
                    html.Div("Best Site", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(best_site or "—", style={"fontSize": "18px", "fontWeight": "700"}),
                ], style={"padding": "10px 14px", "border": "1px solid #eee", "borderRadius": "10px", "flex": "1"}),
                html.Div([
                    html.Div("Suggested Payload Band", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(f"{sweet_lo:,}–{sweet_hi:,} kg", style={"fontSize": "18px", "fontWeight": "700"}),
                ], style={"padding": "10px 14px", "border": "1px solid #eee", "borderRadius": "10px", "flex": "1", "marginLeft": "12px"}),
                html.Div([
                    html.Div("Success @ Band (ALL)", style={"fontSize": "12px", "color": "#666"}),
                    html.Div(f"{sweet_rate*100:.1f}%", style={"fontSize": "18px", "fontWeight": "700"}),
                ], style={"padding": "10px 14px", "border": "1px solid #eee", "borderRadius": "10px", "flex": "1", "marginLeft": "12px"}),
            ],
            style={"display": "flex", "margin": "0 0 12px"}
        ),

        html.Div(
            [
                dcc.Graph(id="success-pie-chart", style={"height": "420px"}),
                dcc.Graph(id="success-payload-scatter", style={"height": "420px"}),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"}
        ),

        # ---------- Methods panel ----------
        html.Details(
            open=False,
            children=[
                html.Summary("Predictive Analysis — Methods (click to expand)"),
                html.Ul([
                    html.Li("Target: first-stage landing success (class = 1)."),
                    html.Li("Features: one-hot encoded LaunchSite & Orbit; numeric PayloadMass; year."),
                    html.Li("Split: stratified train/validation/test with fixed seed."),
                    html.Li("Models compared: Logistic Regression, Decision Tree, SVM, KNN."),
                    html.Li("Tuning: GridSearchCV (k-fold CV), metric = accuracy."),
                    html.Li("Selection: highest test accuracy with stable CV."),
                    html.Li("Evaluation: accuracy, confusion matrix, Precision/Recall/F1 for class = 1."),
                ], style={"lineHeight": "1.6"})
            ],
            style={"marginTop": "14px", "padding": "10px 12px", "border": "1px solid #eee", "borderRadius": "10px", "background": "#fafafa"}
        ),
    ]
)

# ---------- Callbacks ----------
@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def update_pie(site):
    if site == "ALL":
        fig = px.pie(df, names="LaunchSite", values="class", title="Launch Success Distribution by Site")
    else:
        d = df[df["LaunchSite"] == site]
        fig = px.pie(d, names="class", title=f"Launch Site Performance — {site}", hole=0.35)
        fig.update_traces(textinfo="percent+label")
    return fig

@app.callback(Output("success-payload-scatter", "figure"),
              Input("site-dropdown", "value"),
              Input("payload-slider", "value"))
def update_scatter(site, payload_range):
    lo, hi = payload_range
    d = df[(df["PayloadMass"] >= lo) & (df["PayloadMass"] <= hi)]
    title = "Payload Range Success Analysis"
    if site != "ALL":
        d = d[d["LaunchSite"] == site]
        title += f" — {site}"
    fig = px.scatter(
        d, x="PayloadMass", y="class", color="Orbit",
        title=title, labels={"PayloadMass": "Payload (kg)", "class": "Success (1/0)"},
        hover_data=["LaunchSite", "Orbit"]
    )
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Failure", "Success"])
    return fig

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=False)
