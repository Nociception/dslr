import webbrowser
import os

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


def pair_plot(csv_file: str) -> None:
    df = pl.read_csv(csv_file)

    numeric_cols = [
        col for col in df.columns 
        if df[col].dtype in (pl.Float64, pl.Int64) and col != "Index"
    ]

    if not numeric_cols:
        print("No numeric columns found in the dataset.")
        return

    df_sns = df.select(["Hogwarts House"] + numeric_cols).to_pandas()

    sns.set(style="ticks", font_scale=1.0)

    g = sns.pairplot(
        df_sns,
        vars=numeric_cols,
        hue="Hogwarts House",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.4, "s": 20},
        diag_kws={"alpha": 0.5},
        height=3,
        aspect=1.2,
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), rotation=45, ha="right")
            ax.set_ylabel(ax.get_ylabel(), rotation=0)

    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    plt.tight_layout()

    output_png = "pairplot.png"
    g.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Pair plot saved as {output_png}")

    abs_path = os.path.abspath(output_png)
    webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    pair_plot("datasets/dataset_train.csv")
