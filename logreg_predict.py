import sys


def main(test_path: str) -> None:
    pass

if __name__ == "__main__":
    if len(sys.argv) != 2 and sys.argv[1] != "dataset_train.csv".strip():
        print("Usage: ./logreg_train.py dataset_train.csv")
        exit(1)
    main(sys.argv[1])
    print("############## MUST GENERATE HOUSE.CSV ######################")