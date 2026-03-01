import csv
import numpy as np
import random


def minibatch_picker(file_path, features_kept, target, batch_size, mode='train', seed=42):
    random.seed(seed)
    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        
        batch_x, batch_y = [], []

        for row in reader:
            is_val = random.random() < 0.2
            current_line_mode = 'val' if is_val else 'train'

            if current_line_mode != mode:
                continue

            try:
                x = [float(row[col]) if row[col] != "" else 0.0 for col in features_kept]
                batch_x.append(x)
                batch_y.append(row[target])
            except ValueError:
                continue

            if len(batch_x) == batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x, batch_y = [], []

        if batch_x:
            yield np.array(batch_x), np.array(batch_y)
