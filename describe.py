#!.venv/bin/python

import sys

import numpy as np
import polars as pl


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
    assert col_count >= 2, "Can't calculate the std of an empty list or a list with only one element."
    return (ft_sum([(x - col_mean) ** 2 for x in data]) / (col_count))**.5


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


def write_csv(output: dict, functions: list[str], columns: list[str], filename: str) -> None:
    with open(filename, "w") as f:
        f.write("," + ",".join(columns) + "\n")

        for func in functions:
            row = [func]
            for col in columns:
                row.append(f"{output[col][func]:.6f}")
            f.write(",".join(row) + "\n")
    print(f"\n{filename} created ; browse it with a proper solution if you screen is not wide enough.")


def describe(df: pl.DataFrame):
    df = df.drop(["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", "Best Hand"])

    functions = ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    try:
        assert len(df) > 0, "The data is empty."
        output = {}
        for col in df.columns:
            np_col = df[col].to_numpy()
            col_count = ft_count(np_col)
            col_mean = ft_arithmetic_mean(
                np_col,
                col_count=col_count
            )
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
        print("Usage: ./describe.py dataset_[train/test].csv")
        exit(1)
    csv_file = sys.argv[1]
    if not csv_file.endswith(".csv"):
        print("Error: The file is not a csv file")
        exit(1)

    describe(pl.read_csv(csv_file))


if __name__ == "__main__":
    main()
