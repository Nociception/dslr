import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def get_numeric_courses(df: pl.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if df[col].dtype in (pl.Float64, pl.Int64) and col != "Index"
    ]


def get_all_course_values(ctx: pl.SQLContext[pl.LazyFrame], course: str) -> np.ndarray:
    return (
        ctx.execute(f"""
            SELECT
                "{course}"
            FROM
                df
            WHERE
                "{course}" IS NOT NULL
        """)
        .collect()
        .to_numpy()
        .flatten()
    )


def get_house_course_values(
    ctx: pl.SQLContext[pl.LazyFrame], course: str, house: str
) -> np.ndarray:
    return (
        ctx.execute(f"""
            SELECT
                "{course}"
            FROM
                df
            WHERE
                "Hogwarts House" = '{house}'
                AND
                "{course}" IS NOT NULL
        """)
        .collect()
        .to_numpy()
        .flatten()
    )


def compute_proportions(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts / len(values)


def compute_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    return np.linspace(values.min(), values.max(), n_bins + 1)


def plot_course_histogram(
    ax: plt.Axes,
    ctx: pl.SQLContext,
    houses: list[str],
    course: str,
    n_bins: int,
) -> None:
    all_values = get_all_course_values(ctx, course)

    if len(all_values) == 0:
        return

    bin_edges = compute_bin_edges(all_values, n_bins)
    bin_width = bin_edges[1] - bin_edges[0]

    for house in houses:
        values = get_house_course_values(ctx, course, house)

        if len(values) == 0:
            continue

        proportions = compute_proportions(values, bin_edges)
        heights = proportions / bin_width

        ax.bar(
            bin_edges[:-1],
            heights,
            width=bin_width,
            align="edge",
            alpha=0.4,
            label=house,
        )

    ax.set_title(course)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")


def histogram_all_courses(df: pl.DataFrame) -> None:
    ctx = pl.SQLContext()
    ctx.register("df", df)

    courses = get_numeric_courses(df)
    houses = df["Hogwarts House"].unique().to_list()

    n_bins = 30
    n_cols = 4
    n_rows = math.ceil((len(courses) + 1) / n_cols)

    mosaic = []
    idx = 0
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            if idx < len(courses):
                row.append(str(idx))
            elif idx == len(courses):
                row.append("legend")
            else:
                row.append(".")
            idx += 1
        mosaic.append(row)

    _, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(20, 4 * n_rows),
        constrained_layout=True,
    )

    legend_handles = None
    legend_labels = None

    for i, course in enumerate(courses):
        ax = axes[str(i)]

        plot_course_histogram(
            ax=ax,
            ctx=ctx,
            houses=houses,
            course=course,
            n_bins=n_bins,
        )

        handles, labels = ax.get_legend_handles_labels()

        if legend_handles is None and handles:
            legend_handles = handles
            legend_labels = labels

        ax.legend().remove()

    legend_ax = axes["legend"]
    legend_ax.axis("off")

    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=False,
        ncol=1,
    )

    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("datasets/dataset_train.csv")
    histogram_all_courses(df)
    print("Include best hand histogram")
