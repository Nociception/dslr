import sys
import numpy as np
import polars as pl

def ft_count(data):
    return (len([x for x in data if not np.isnan(x)]))

def ft_nan_count(data):
    return (len([x for x in data if np.isnan(x)]))

def ft_mean(data):
    return (np.sum([x for x in data if not np.isnan(x)]) / ft_count(data))

def ft_std(data):
    mean = ft_mean(data)
    return (np.sqrt(np.sum([(x - mean) ** 2 for x in data if not np.isnan(x)]) / ft_count(data)))

def describe(df):
    # count ; mean ; std ; min ; 25% ; 50% ; 75% ; max

    df = df.drop(["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    
    print("\t\t\t", [col + "\t" for col in df.columns])

    output = {}
    for col in df.columns:
        output[col] = {
            "count: ": ft_count(df[col].to_numpy()),
            "null_count: ": ft_nan_count(df[col].to_numpy()),
            "mean: ": ft_mean(df[col].to_numpy()),
            "std: ": ft_std(df[col].to_numpy()),
            
        }


    return

def main():
    # if sys.argv[2][-4] != ".csv":
    #     print("Error: The file is not a csv file")
    #     exit(1)
    # csv_file = sys.argv[2]

    # df_train = pl.read_csv(csv_file)
    # describe(df_train)
    df = pl.read_csv("datasets/dataset_train.csv")
    df = df.drop(["Hogwarts House", "Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    print(df.describe())

    data = np.array([1, 2, np.nan, 4, 5, np.nan])
    print(ft_std(data))
    print(np.nanstd(data))

if __name__ == "__main__":
    main()