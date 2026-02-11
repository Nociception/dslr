import polars as pl
from describe import describe


def main():
    df_train = pl.read_csv("datasets/dataset_train.csv")

    describe(df_train)


if __name__ == "__main__":
    main()
