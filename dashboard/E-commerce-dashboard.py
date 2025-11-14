#!/usr/bin/env python3
"""
Plotly Dash dashboard that surfaces key e-commerce pricing KPIs and trends.

The app looks for curated CSV exports inside ``data/processed`` (falling back to
``data/raw``). If no substantive file is present it will synthesize a small,
repeatable sample dataset so that the dashboard remains interactive even in a
fresh clone of the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


DATA_CANDIDATES: Tuple[Path, ...] = (
    Path("data/processed/ebay_cleaned_with_extracted_brands.csv"),
    Path("data/raw/ebay_cleaned_dataset.csv"),
)


def load_data() -> pd.DataFrame:
    """Load the richest available CSV or fall back to a synthetic sample."""

    for candidate in DATA_CANDIDATES:
        if candidate.exists() and candidate.stat().st_size > 128:
            df = pd.read_csv(candidate)
            df["source_file"] = candidate.name
            break
    else:
        df = _build_mock_data()
        df["source_file"] = "synthetic_seed.csv"

    rename_map = {
        "category_code": "category",
        "category_name": "category",
        "brand_name": "brand",
        "price_source": "platform",
        "seller_name": "seller",
    }
    df = df.rename(columns=rename_map)

    df["platform"] = df.get("platform", "Marketplace").fillna("Marketplace")
    df["brand"] = df.get("brand", "Unknown Brand").fillna("Unknown Brand")
    df["category"] = df.get("category", "Uncategorized").fillna("Uncategorized")
    df["event_type"] = df.get("event_type", "view").fillna("view")
    df["seller"] = df.get("seller", "Unknown Seller").fillna("Unknown Seller")

    if "price" not in df.columns:
        df["price"] = np.random.default_rng(42).uniform(5, 500, len(df))
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "shipping_price" not in df.columns:
        df["shipping_price"] = df["price"] * 0.05
    df["shipping_price"] = pd.to_numeric(df["shipping_price"], errors="coerce")

    if "ratings" not in df.columns:
        df["ratings"] = np.random.default_rng(7).uniform(3, 5, len(df))
    df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    else:
        df["event_time"] = pd.date_range(
            end=pd.Timestamp.utcnow(), periods=len(df), freq="H"
        )

    df["event_date"] = df["event_time"].dt.date
    df = df.dropna(subset=["price", "event_time"]).reset_index(drop=True)
    return df


def _build_mock_data(rows: int = 2500) -> pd.DataFrame:
    """Create a deterministic synthetic dataset for local experimentation."""

    rng = np.random.default_rng(123)
    categories = [
        "smartphone",
        "tablet",
        "laptop",
        "audio.headphone",
        "wearable",
    ]
    brands = [
        "Apple",
        "Samsung",
        "Sony",
        "Google",
        "Fitbit",
        "Bose",
        "Lenovo",
    ]
    platforms = ["eBay", "Kaggle"]
    sellers = ["Best Deals", "TechWorld", "PrimeElectronics", "Marketplace Pro"]

    start = pd.Timestamp.utcnow() - pd.Timedelta(days=180)
    event_times = start + pd.to_timedelta(rng.integers(0, 180 * 24, size=rows), unit="h")

    data = {
        "event_time": event_times,
        "product_id": rng.choice(range(10_000, 99_999), size=rows, replace=False),
        "category": rng.choice(categories, size=rows),
        "brand": rng.choice(brands, size=rows),
        "platform": rng.choice(platforms, size=rows, p=[0.6, 0.4]),
        "seller": rng.choice(sellers, size=rows),
        "event_type": rng.choice(
            ["view", "cart", "purchase"], size=rows, p=[0.55, 0.25, 0.20]
        ),
    }

    base_prices = {
        "smartphone": 650,
        "tablet": 420,
        "laptop": 900,
        "audio.headphone": 180,
        "wearable": 230,
    }

    prices = []
    for cat, platform in zip(data["category"], data["platform"]):
        noise = rng.normal(0, base_prices[cat] * 0.1)
        platform_bias = -25 if platform == "eBay" else 25
        price = max(15, base_prices[cat] + noise + platform_bias)
        prices.append(price)

    data["price"] = np.round(prices, 2)
    data["shipping_price"] = np.round(
        np.clip(rng.normal(10, 3, size=rows), 0, None), 2
    )
    data["ratings"] = np.round(rng.uniform(3.3, 4.9, size=rows), 2)
    data["title"] = [
        f"{brand} {cat.split('.')[0].title()} Model {rng.integers(10, 99)}"
        for brand, cat in zip(data["brand"], data["category"])
    ]
    return pd.DataFrame(data)


def build_kpi_cards(df: pd.DataFrame) -> List[html.Div]:
    """Create KPI cards for the filtered dataset."""

    if df.empty:
        metrics: Iterable[Tuple[str, str]] = [("No results", "—")]
    else:
        metrics = [
            ("Listings", f"{len(df):,}"),
            ("Avg. Price", f"${df['price'].mean():,.2f}"),
            ("Median Price", f"${df['price'].median():,.2f}"),
            ("Unique Brands", f"{df['brand'].nunique():,}"),
            ("Avg. Rating", f"{df['ratings'].mean():.2f} ★"),
            (
                "Conversion %",
                f"{(df['event_type'].eq('purchase').mean() * 100):.1f}%",
            ),
        ]

    cards = []
    for label, value in metrics:
        cards.append(
            html.Div(
                [
                    html.P(label, className="kpi-label"),
                    html.H3(value, className="kpi-value"),
                ],
                className="kpi-card",
            )
        )
    return cards


def build_price_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart showing daily price trends per platform."""

    if df.empty:
        return px.line(title="Price Trend (no data)")

    trend = (
        df.groupby(["event_date", "platform"])
        .agg(avg_price=("price", "mean"), orders=("event_type", "count"))
        .reset_index()
    )
    fig = px.line(
        trend,
        x="event_date",
        y="avg_price",
        color="platform",
        markers=True,
        title="Average Price Trend by Platform",
        labels={"event_date": "Date", "avg_price": "Average Price (USD)"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def build_brand_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart ranking brands by revenue-like metric."""

    if df.empty:
        return px.bar(title="Top Brands (no data)")

    top_brands = (
        df.groupby("brand")
        .agg(
            avg_price=("price", "mean"),
            records=("price", "count"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
        .head(10)
    )
    fig = px.bar(
        top_brands,
        x="brand",
        y="records",
        color="avg_price",
        color_continuous_scale="Tealrose",
        title="Top 10 Brands by Listings",
        labels={"records": "Listings", "brand": "Brand"},
    )
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def build_category_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap showing category share."""

    if df.empty:
        return px.treemap(title="Category Mix (no data)")

    mix = (
        df.groupby(["category", "platform"])
        .size()
        .reset_index(name="count")
    )
    fig = px.treemap(
        mix,
        path=["platform", "category"],
        values="count",
        title="Category Mix by Platform",
    )
    return fig


def build_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Price distribution per brand."""

    if df.empty:
        return px.box(title="Price Distribution (no data)")

    top_brands = df["brand"].value_counts().head(6).index
    subset = df[df["brand"].isin(top_brands)]
    fig = px.box(
        subset,
        x="brand",
        y="price",
        color="brand",
        title="Price Distribution for Top Brands",
        points="all",
    )
    fig.update_layout(showlegend=False)
    return fig


def filter_frame(
    df: pd.DataFrame,
    categories: List[str] | None,
    brands: List[str] | None,
    platforms: List[str] | None,
    price_range: List[float],
    event_types: List[str] | None,
) -> pd.DataFrame:
    """Apply all interactive filters to the dataset."""

    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]
    if platforms:
        filtered = filtered[filtered["platform"].isin(platforms)]
    if event_types:
        filtered = filtered[filtered["event_type"].isin(event_types)]
    if price_range:
        filtered = filtered[
            filtered["price"].between(price_range[0], price_range[1], inclusive="both")
        ]
    return filtered


def serve_layout(df: pd.DataFrame) -> html.Div:
    """Compose the dashboard layout."""

    return html.Div(
        [
            html.H1("E-commerce Price Trends Dashboard", className="title"),
            html.P(
                [
                    "Source file: ",
                    html.Strong(df["source_file"].iloc[0]),
                    " — Drag the controls below to explore KPIs, compare brands, ",
                    "and spot price anomalies across platforms.",
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Category"),
                            dcc.Dropdown(
                                id="category-filter",
                                options=[
                                    {"label": c.title(), "value": c}
                                    for c in sorted(df["category"].unique())
                                ],
                                multi=True,
                            ),
                        ],
                        className="filter-card",
                    ),
                    html.Div(
                        [
                            html.Label("Brand"),
                            dcc.Dropdown(
                                id="brand-filter",
                                options=[
                                    {"label": b, "value": b}
                                    for b in sorted(df["brand"].unique())
                                ],
                                multi=True,
                            ),
                        ],
                        className="filter-card",
                    ),
                    html.Div(
                        [
                            html.Label("Platform"),
                            dcc.Checklist(
                                id="platform-filter",
                                options=[
                                    {"label": p, "value": p}
                                    for p in sorted(df["platform"].unique())
                                ],
                                value=sorted(df["platform"].unique()),
                                inline=True,
                            ),
                        ],
                        className="filter-card",
                    ),
                    html.Div(
                        [
                            html.Label("Event Type"),
                            dcc.Checklist(
                                id="event-filter",
                                options=[
                                    {"label": e.title(), "value": e}
                                    for e in sorted(df["event_type"].unique())
                                ],
                                value=sorted(df["event_type"].unique()),
                                inline=True,
                            ),
                        ],
                        className="filter-card",
                    ),
                ],
                className="filters",
            ),
            html.Div(
                [
                    html.Label("Price Range (USD)"),
                    dcc.RangeSlider(
                        id="price-slider",
                        min=float(df["price"].min()),
                        max=float(df["price"].max()),
                        step=5,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                        value=[
                            float(df["price"].quantile(0.05)),
                            float(df["price"].quantile(0.95)),
                        ],
                    ),
                ],
                className="slider-card",
            ),
            html.Div(id="kpi-cards", className="kpi-row"),
            html.Div(
                [
                    dcc.Graph(id="price-trend"),
                    dcc.Graph(id="brand-bar"),
                ],
                className="chart-grid",
            ),
            html.Div(
                [
                    dcc.Graph(id="category-treemap"),
                    dcc.Graph(id="price-distribution"),
                ],
                className="chart-grid",
            ),
        ],
        className="container",
    )


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Wire Dash callbacks for interactivity."""

    @app.callback(
        Output("price-trend", "figure"),
        Output("brand-bar", "figure"),
        Output("category-treemap", "figure"),
        Output("price-distribution", "figure"),
        Output("kpi-cards", "children"),
        Input("category-filter", "value"),
        Input("brand-filter", "value"),
        Input("platform-filter", "value"),
        Input("price-slider", "value"),
        Input("event-filter", "value"),
    )
    def _update_charts(
        categories, brands, platforms, price_range, event_types
    ):
        filtered = filter_frame(
            df,
            categories,
            brands,
            platforms,
            price_range,
            event_types,
        )
        return (
            build_price_trend(filtered),
            build_brand_bar(filtered),
            build_category_treemap(filtered),
            build_price_distribution(filtered),
            build_kpi_cards(filtered),
        )


def main() -> None:
    df = load_data()
    app = Dash(__name__, title="Price Trends Dashboard")
    app.layout = serve_layout(df)
    register_callbacks(app, df)
    app.run(debug=True)


if __name__ == "__main__":
    main()
