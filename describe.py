#!.venv/bin/python

import sys

import numpy as np
import polars as pl
import numpy.typing as npt


def ft_sum(arr: list[int | float]) -> float:
    return sum(value for value in arr if not np.isnan(value))


def ft_count(data: np.ndarray) -> int:
    return len([x for x in data if not np.isnan(x)])


def ft_nan_count(data: np.ndarray) -> int:
    return len([x for x in data if np.isnan(x)])


def ft_arithmetic_mean(
    data: np.ndarray,
    col_count: int | None = None,
) -> float:
    assert col_count is not None, "ft_arithmetic_mean did not receive col_count arg."
    assert col_count > 0, "Can't calculate the mean of an empty list"
    return ft_sum([x for x in data]) / col_count


def ft_std(
    data: np.ndarray,
    col_count: int = -1,
    col_mean: float | None = None,
) -> float:
    assert col_count != -1, "ft_std did not receive col_count arg."
    assert col_count is not None, "ft_std did not receive col_mean arg."
    assert col_count >= 2, (
        "Can't calculate the std of an empty list or a list with only one element."
    )
    return (ft_sum([(x - col_mean) ** 2 for x in data]) / (col_count)) ** 0.5


def ft_min(data: np.ndarray) -> int | float:
    arr = [x for x in data if not np.isnan(x)]
    min = arr[0]
    for x in arr:
        if x < min:
            min = x
    return min


def ft_max(data: np.ndarray) -> int | float:
    arr = [x for x in data if not np.isnan(x)]
    max = arr[0]
    for x in arr:
        if x > max:
            max = x
    return max


def ft_percentile(data: np.ndarray, p: float) -> int | float:
    assert p >= 0 and p <= 1, "The percentile must be between 0 and 1."
    arr = [x for x in data if not np.isnan(x)]
    arr.sort()
    return arr[int(len(arr) * p)]


def write_csv(
    output: dict, functions: list[str], columns: list[str], filename: str
) -> None:
    with open(filename, "w") as f:
        f.write("," + ",".join(columns) + "\n")

        for func in functions:
            row = [func]
            for col in columns:
                row.append(f"{output[col][func]:.6f}")
            f.write(",".join(row) + "\n")
    print(
        f"\n{filename} created ; browse it with a proper solution if your screen is not wide enough."
    )


def ft_anova_f_scores(
    X: npt.NDArray[np.float64],  # shape (m, n)
    y: npt.NDArray[np.str_],  # shape (m,)
) -> npt.NDArray[np.float64]:
    n = X.shape[1]
    f_scores = np.zeros(n, dtype=np.float64)

    for j in range(n):
        mask_valid = ~np.isnan(X[:, j])
        X_j = X[mask_valid, j]
        y_j = y[mask_valid]

        m_j = len(X_j)
        classes = np.unique(y_j)
        K = len(classes)

        if m_j <= K:
            f_scores[j] = np.nan
            continue

        global_mean = np.mean(X_j)
        ss_between = 0.0
        ss_within = 0.0

        for c in classes:
            class_mask = y_j == c
            X_c = X_j[class_mask]
            n_k = len(X_c)
            if n_k == 0:
                continue

            mean_k = np.mean(X_c)
            ss_between += n_k * (mean_k - global_mean) ** 2
            ss_within += np.sum((X_c - mean_k) ** 2)

        ms_between = ss_between / (K - 1)
        ms_within = ss_within / (m_j - K)

        if ms_within == 0:
            f_scores[j] = 0.0
        else:
            f_scores[j] = ms_between / ms_within

    return f_scores


def describe(df: pl.DataFrame):
    df_dup = df
    df = df.drop(
        ["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    )

    functions = [
        "count",
        "null_count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "anova_f_score",
    ]
    try:
        assert len(df) > 0, "The data is empty."
        output = {}
        for col in df.columns:
            np_col = df[col].to_numpy()
            col_count = ft_count(np_col)
            col_mean = ft_arithmetic_mean(np_col, col_count=col_count)
            output[col] = {
                "count": col_count,
                "null_count": ft_nan_count(np_col),
                "mean": col_mean,
                "std": ft_std(
                    np_col,
                    col_count=col_count,
                    col_mean=col_mean,
                ),
                "min": ft_min(np_col),
                "25%": ft_percentile(np_col, 0.25),
                "50%": ft_percentile(np_col, 0.50),
                "75%": ft_percentile(np_col, 0.75),
                "max": ft_max(np_col),
                "anova_f_score": ft_anova_f_scores(
                    X=df_dup.select(pl.col(col)).to_numpy(),  # shape (m, 1)
                    y=df_dup.select(pl.col("Hogwarts House"))
                    .to_numpy()
                    .flatten(),  # shape (m,)
                )[0],
            }
    except AssertionError as e:
        print("Error: ", e)
        exit(1)

    print(f"{'':<12}", end="")
    for col in df.columns:
        print(f"{col:<12}", end="")
    print()
    for f in functions:
        print(f"{f:<12}", end="")
        for col in df.columns:
            print(f"{output[col][f]:<12.2f}", end="")
        print()

    write_csv(
        output=output,
        functions=functions,
        columns=df.columns,
        filename="describe.csv",
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: ./describe.py datasets/dataset_[train/test].csv")
        exit(1)
    csv_file = sys.argv[1]
    if not csv_file.endswith(".csv"):
        print("Error: The file is not a csv file")
        exit(1)

    describe(pl.read_csv(csv_file))


if __name__ == "__main__":
    main()
