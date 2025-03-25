import sys
import numpy as np
import polars as pl

def ft_count(data):
    return (len([x for x in data if not np.isnan(x)]))

def ft_nan_count(data):
    return (len([x for x in data if np.isnan(x)]))

def ft_mean(data):
    assert ft_count(data) > 0, "Can't calculate the mean of an empty list"
    return (np.sum([x for x in data if not np.isnan(x)]) / ft_count(data))

def ft_std(data):
    assert ft_count(data) >= 2, "Can't calculate the std of an empty list or a list with only one element"
    mean = ft_mean(data)
    return (np.sqrt(np.sum([(x - mean) ** 2 for x in data if not np.isnan(x)]) / (ft_count(data) - 1)))

def ft_min(data):
    arr = [x for x in data if not np.isnan(x)]
    min = arr[0]
    for x in arr:
        if x < min:
            min = x
    return (min)

def ft_max(data):
    arr = [x for x in data if not np.isnan(x)]
    max = arr[0]
    for x in arr:
        if x > max:
            max = x
    return (max)

def ft_percentile(data, p):
    assert p >= 0 and p <= 1, "The percentile must be between 0 and 1"
    arr = [x for x in data if not np.isnan(x)]
    arr.sort()
    return (arr[int(len(arr) * p)])


def describe(df):
    # count ; mean ; std ; min ; 25% ; 50% ; 75% ; max

    df = df.drop(["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    functions = ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    try:
        assert len(df) > 0, "The data is empty"
        output = {}
        for col in df.columns:
            output[col] = {
                "count": ft_count(df[col].to_numpy()),
                "null_count": ft_nan_count(df[col].to_numpy()),
                "mean": ft_mean(df[col].to_numpy()),
                "std": ft_std(df[col].to_numpy()),
                "min": ft_min(df[col].to_numpy()),
                "25%": ft_percentile(df[col].to_numpy(), 0.25),
                "50%": ft_percentile(df[col].to_numpy(), 0.50),
                "75%": ft_percentile(df[col].to_numpy(), 0.75),
                "max": ft_max(df[col].to_numpy()),
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

def main():
    if len(sys.argv) != 3:
        print("Error: Wrong number of arguments")
        exit(1)
    if sys.argv[2][-4] != ".csv":
        print("Error: The file is not a csv file")
        exit(1)
    csv_file = sys.argv[2]

    df_train = pl.read_csv(csv_file)
    describe(df_train)

if __name__ == "__main__":
    main()
