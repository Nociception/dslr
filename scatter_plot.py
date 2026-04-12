import itertools
import webbrowser
from threading import Timer

import polars as pl
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


df = pl.read_csv("datasets/dataset_train.csv")

numeric_cols = [
    col
    for col in df.columns
    if df[col].dtype in (pl.Float64, pl.Int64) and col != "Index"
]

pairs = []

for col1, col2 in itertools.combinations(numeric_cols, 2):
    corr = df.select(pl.corr(col1, col2)).item()
    if corr is not None:
        pairs.append((col1, col2, corr))

houses = df.select("Hogwarts House").drop_nulls().unique().to_series().to_list()


def create_corr_figure() -> go.Figure:

    x_vals = []
    y_vals = []
    colors = []
    hover_text = []

    for col1, col2, corr in pairs:
        x_vals.append(col1)
        y_vals.append(col2)
        colors.append(corr)
        hover_text.append(f"{col1} vs {col2}<br>corr = {corr:.4f}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=12,
                color=colors,
                colorscale="RdBu",
                cmin=-1,
                cmax=1,
                colorbar=dict(
                    title="Correlation",
                    x=1.02,
                ),
            ),
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="All Feature Correlations",
        margin=dict(l=40, r=40, t=60, b=40),
        dragmode=False,
    )

    return fig


def create_scatter(col1: str, col2: str) -> go.Figure:

    df_clean = df.select(["Hogwarts House", col1, col2]).drop_nulls().to_pandas()

    fig = go.Figure()

    for house in houses:
        df_house = df_clean[df_clean["Hogwarts House"] == house]

        fig.add_trace(
            go.Scatter(
                x=df_house[col1],
                y=df_house[col2],
                mode="markers",
                name=house,
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"{col1} vs {col2}",
        xaxis_title=col1,
        yaxis_title=col2,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="correlation-map",
                    figure=create_corr_figure(),
                    style={"width": "48%", "display": "inline-block"},
                ),
                dcc.Graph(
                    id="scatter-plot",
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    "Click on a correlation point to display the corresponding scatter plot.",
                    style={
                        "textAlign": "left",
                        "marginTop": "10px",
                        "fontStyle": "italic",
                        "color": "gray",
                    },
                ),
            ]
        )
    ]
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("correlation-map", "clickData"),
)
def update_scatter(clickData) -> go.Figure:

    if clickData is None:
        col1, col2, _ = pairs[0]
    else:
        point_index = clickData["points"][0]["pointIndex"]
        col1, col2, _ = pairs[point_index]

    return create_scatter(col1, col2)


if __name__ == "__main__":
    port = 8050
    url = f"http://127.0.0.1:{port}/"

    Timer(1, lambda: webbrowser.open(url)).start()

    app.run(debug=True, use_reloader=False, port=port)
