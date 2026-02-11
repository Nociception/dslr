import math
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


def histogram(df: pl.DataFrame) -> None:
    numeric_cols = [
        col for col in df.columns
        if df[col].dtype in (pl.Float64, pl.Int64)
        and col not in ["Index"]
    ]

    df_pd = df.select(["Hogwarts House"] + numeric_cols).to_pandas()

    n_cols = 4
    n_rows = math.ceil((len(numeric_cols) + 1) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(20, 4 * n_rows),
        constrained_layout=True
    )

    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, course in enumerate(numeric_cols):
        ax = axes[i]

        sns.histplot(
            data=df_pd,
            x=course,
            hue="Hogwarts House",
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.4,
            bins=30,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            if legend_handles is None:
                legend_handles = legend.legend_handles
                legend_labels = [t.get_text() for t in legend.texts]

            legend.remove()

    legend_ax = axes[len(numeric_cols)]
    legend_ax.axis("off")

    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=False,
        ncol=1,
    )

    for j in range(len(numeric_cols) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()



if __name__ == "__main__":
    df = pl.read_csv("datasets/dataset_train.csv")
    histogram(df)
