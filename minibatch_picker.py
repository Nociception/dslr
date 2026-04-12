import csv
import numpy as np
import random
from typing import List, Generator, Tuple


def minibatch_picker(
    file_path: str,
    features_kept: List[str],
    target: str,
    batch_size: int,
    mode: str = "train",
    shuffle_within_batch: bool = False,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

    random.seed(seed)

    with open(file_path, mode="r") as f:
        reader = csv.DictReader(f)

        batch_x, batch_y = [], []

        for row in reader:
            is_val = random.random() < 0.2
            current_line_mode = "val" if is_val else "train"

            if current_line_mode != mode:
                continue

            try:
                x = [
                    float(row[col]) if row[col] != "" else 0.0 for col in features_kept
                ]
                batch_x.append(x)
                batch_y.append(row[target])
            except ValueError:
                continue

            if len(batch_x) == batch_size:
                X = np.array(batch_x, dtype=np.float64)
                y = np.array(batch_y)

                if shuffle_within_batch:
                    perm = np.random.permutation(len(X))
                    X = X[perm]
                    y = y[perm]

                yield X, y
                batch_x, batch_y = [], []

        if batch_x:
            X = np.array(batch_x, dtype=np.float64)
            y = np.array(batch_y)

            if shuffle_within_batch:
                perm = np.random.permutation(len(X))
                X = X[perm]
                y = y[perm]

            yield X, y
